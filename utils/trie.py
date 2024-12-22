from typing import Dict
# Disclaimer: Trie class made by ChatGPT. 
# If needed MARISA-based tries should be used; for now this seems like overkill.
#SEE https://github.com/pytries/marisa-trie 

class TrieNode:
    def __init__(self):
        self.children: Dict[str, TrieNode] = {}
        self.is_end_of_uuid: bool = False


class UUIDv4Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, uuid_str: str) -> None:
        """
        Inserts a UUIDv4 string into the trie.
        """
       # if not self.is_valid_uuidv4(uuid_str):
        #    raise ValueError(f"{uuid_str} is not a valid UUIDv4.")
        
        node = self.root
        for char in uuid_str:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end_of_uuid = True

    def search(self, uuid_str: str) -> bool:
        """
        Searches for a UUIDv4 string in the trie.
        Returns True if found, otherwise False.
        """
        node = self.root
        for char in uuid_str:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end_of_uuid
