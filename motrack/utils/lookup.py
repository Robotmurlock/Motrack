"""
Dynamic LookupTable (class mappings).
"""
from typing import Dict, Optional, Iterable, Tuple, List


class LookupTable:
    """Vocabulary - Label lookup table (token <-> index)."""

    def __init__(
            self,
            token_to_index: Optional[Dict[str, int]] = None,
            unknown_token: str = '<unk>',
            add_unknown_token: bool = True
    ):
        """
        Args:
            token_to_index: Predefine token to index mappings.
            unknown_token: Unknown token value.
            add_unknown_token: Use unknown token.
        """
        self._token_to_index = token_to_index
        self._unknown_token = unknown_token
        self._add_unknown_token = add_unknown_token

        if self._token_to_index is None:
            self._token_to_index = {}

        if unknown_token not in self._token_to_index and add_unknown_token:
            self._token_to_index[unknown_token] = len(self._token_to_index)

        self._index_to_token = {v: k for k, v in self._token_to_index.items()}
        self._next_index = len(self._token_to_index)

    def add(self, token: str) -> int:
        """
        Adds token to the lookup table if it does not already exist.

        Args:
            token: Label (token)

        Returns:
            label (token) index
        """
        if token in self._token_to_index:
            return self._token_to_index[token]

        new_index = self._next_index
        self._next_index += 1
        self._token_to_index[token] = new_index
        self._index_to_token[new_index] = token
        return new_index

    def lookup(self, token: str) -> int:
        """
        Acquires token index if it exists in the table.
        In case the token does not exist in the lookup table:
            - and unknown token is used then unknown token index is returned;
            - otherwise KeyError is raised

        Raises:
            KeyError: Unknown token

        Returns:
            label (token) index
        """
        if token not in self._token_to_index and self._add_unknown_token:
            return self._token_to_index[self._unknown_token]

        return self._token_to_index[token]

    def inverse_lookup(self, index: int) -> str:
        """
        Inverse to `lookup`. Acquire token by index.

        Raises:
            KeyError: Unknown index

        Returns:
            label (token)
        """
        return self._index_to_token[index]

    def __iter__(self) -> Iterable[Tuple[str, int]]:
        for token, index in self._token_to_index.items():
            yield token, index

    @property
    def tokens(self) -> List[str]:
        """
        Returns:
            List of tokens in lookup table
        """
        return [token for token, _ in self]

    def __getitem__(self, token: str) -> int:
        return self.lookup(token)  # Alias for `lookup`

    def __len__(self):
        return self._next_index

    def serialize(self) -> dict:
        return {
            'token_to_index': self._token_to_index,
            'unknown_token': self._unknown_token,
            'add_unknown_token': self._add_unknown_token
        }

    @classmethod
    def deserialize(cls, raw: dict) -> 'LookupTable':
        return cls(**raw)


def run_test() -> None:
    lookup = LookupTable()
    lookup.add('dog')
    lookup.add('cat')

    assert lookup['dog'] == lookup.lookup('dog') == 1
    assert lookup['cat'] == lookup.lookup('cat') == 2
    assert lookup.inverse_lookup(1) == 'dog'
    assert lookup.inverse_lookup(2) == 'cat'
    assert lookup.tokens == ['<unk>', 'dog', 'cat']

    deserialized = LookupTable.deserialize(lookup.serialize())

    assert deserialized['dog'] == deserialized.lookup('dog') == 1
    assert deserialized['cat'] == deserialized.lookup('cat') == 2
    assert deserialized.inverse_lookup(1) == 'dog'
    assert deserialized.inverse_lookup(2) == 'cat'


if __name__ == '__main__':
    run_test()
