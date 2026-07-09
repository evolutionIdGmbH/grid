"""GridLogitsProcessor: the GRID tool-family logits-processor shape (DESIGN.md SS4.3, E13).

- Array-type normalization (torch/numpy/list/tuple; mlx/jax via guarded imports).
- ``_seq_start_idx`` anchors the constrained span on first call (prompt ids are
  never constrained).
- State registry: splitmix64
  rolling keys with (n_generated, last_token) validation and refold-on-mismatch
  (beam/batch reorder tolerance without Theta(n) hashing per step).
- Write in processor-only mode degrades to a singleton mask (forced_ids[0]);
  a processor must never union a whole Write span into one step's mask (SS4.5).
- Lifecycle: FRESH -> ANCHORED -> FINISHED via the statechart; ``finish()`` is
  called by the adapter on ANY stop; reuse raises ProcessorReuseError.
  ``copy.copy`` == ``.copy()`` via ``__copy__`` (fresh guide, fresh registry).
"""

from __future__ import annotations

import torch

from grid._statecharts.engine import Statechart, load_chart
from grid.errors import ProcessorReuseError
from grid.guide import COMPLETE, GridGuide, GridState
from grid.protocols import Generate, Write

_M64 = (1 << 64) - 1


def _mix(key: int, token_id: int) -> int:
    x = (key ^ (token_id * 0x9E3779B97F4A7C15)) & _M64
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & _M64
    return x ^ (x >> 31)


_EMPTY_KEY = 0x9E3779B97F4A7C15


class GridBaseLogitsProcessor:
    """Base processor: normalize array types, guarantee 2D, delegate to process_logits."""

    def process_logits(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def __call__(self, input_ids, logits):
        torch_logits = self._to_torch(logits)
        torch_ids = self._to_torch(input_ids)
        if torch_logits.dim() == 2:
            out = self.process_logits(torch_ids, torch_logits)
        else:
            out = self.process_logits(torch_ids.unsqueeze(0), torch_logits.unsqueeze(0)).squeeze(0)
        return self._from_torch(out, type(logits))

    @staticmethod
    def _to_torch(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        try:
            import numpy as np

            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
        except ImportError:  # pragma: no cover
            pass
        if isinstance(x, (list, tuple)):
            return torch.tensor(x)
        raise TypeError(f"unsupported logits type {type(x)}")

    @staticmethod
    def _from_torch(t: torch.Tensor, target: type):
        if target is torch.Tensor:
            return t
        try:
            import numpy as np

            if target is np.ndarray:
                return t.detach().numpy()
        except ImportError:  # pragma: no cover
            pass
        if target is list:
            return t.detach().tolist()
        if target is tuple:
            return tuple(t.detach().tolist())
        raise TypeError(f"unsupported target type {target}")


class GridLogitsProcessor(GridBaseLogitsProcessor):
    def __init__(self, tokenizer, guide: GridGuide) -> None:
        self.tokenizer = tokenizer
        self.guide = guide
        self._sc = Statechart(load_chart("logits_processor"))
        self._seq_start_idx: int | None = None
        self._guide_states: dict[int, GridState] = {_EMPTY_KEY: guide.initial_state}
        self._key_meta: dict[int, tuple[int, int | None]] = {_EMPTY_KEY: (0, None)}
        self._row_cursors: dict[int, tuple[int, int]] = {}  # row -> (key, gen_len)

    # -- E13 lifecycle -------------------------------------------------------

    def finish(self) -> None:
        if self._sc.state != "FINISHED":
            self._sc.fire("finish")

    def copy(self) -> GridLogitsProcessor:
        return GridLogitsProcessor(self.tokenizer, self.guide.copy())

    def __copy__(self) -> GridLogitsProcessor:
        return self.copy()

    # -- state registry ------------------------------------------------------

    def _fold(self, gen: list[int]) -> int:
        key = _EMPTY_KEY
        for t in gen:
            key = _mix(key, t)
        return key

    def _state_for(self, row: int, gen: list[int]) -> GridState:
        if not gen:
            self._row_cursors[row] = (_EMPTY_KEY, 0)
            return self._guide_states[_EMPTY_KEY]
        cursor = self._row_cursors.get(row)
        if cursor and cursor[1] == len(gen) - 1:
            key = _mix(cursor[0], gen[-1])
        else:
            key = self._fold(gen)
        meta = self._key_meta.get(key)
        if meta is not None and meta != (len(gen), gen[-1]):
            key = self._fold(gen)  # 64-bit collision or reorder: refold, treat as miss
        if key not in self._guide_states:
            prev_key = self._fold(gen[:-1])
            if prev_key not in self._guide_states:
                prev = self.guide.initial_state
                for t in gen[:-1]:
                    prev = self.guide.get_next_state(prev, t)
                self._guide_states[prev_key] = prev
                self._key_meta[prev_key] = (len(gen) - 1, gen[-2] if len(gen) > 1 else None)
            state = self.guide.get_next_state(self._guide_states[prev_key], gen[-1])
            self._guide_states[key] = state
            self._key_meta[key] = (len(gen), gen[-1])
        self._row_cursors[row] = (key, len(gen))
        return self._guide_states[key]

    # -- hot path --------------------------------------------------------------

    def process_logits(self, input_ids: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
        if self._sc.state == "FINISHED":
            raise ProcessorReuseError("processor is single-use per generation (E13)")
        self._sc.fire("first_call" if self._sc.state == "FRESH" else "step")
        if self._seq_start_idx is None:
            self._seq_start_idx = len(input_ids[0])

        all_complete = True
        for i in range(len(input_ids)):
            gen = [int(t) for t in input_ids[i][self._seq_start_idx:]]
            state = self._state_for(i, gen)
            if state.status != COMPLETE:
                all_complete = False
            instr = self.guide.get_next_instruction(state)
            if isinstance(instr, Generate):
                if instr.tokens is None:
                    continue  # protocol: Generate(None) = unconstrained row; skip masking
                allowed = torch.as_tensor(instr.tokens, dtype=torch.long)
            else:
                assert isinstance(instr, Write)
                allowed = torch.as_tensor(instr.tokens, dtype=torch.long)[:1]  # SS4.5 mode 2
            mask = torch.ones_like(logits[i], dtype=torch.bool)
            mask[allowed] = False
            logits[i] = logits[i].masked_fill(mask, float("-inf"))

        if all_complete and len(input_ids) > 0:
            self._sc.fire("all_complete")
        return logits
