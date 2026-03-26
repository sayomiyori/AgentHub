from langchain_text_splitters import RecursiveCharacterTextSplitter


class TextChunker:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""],
        )

    def split(self, text: str) -> list[str]:
        if not text.strip():
            return []
        return self.splitter.split_text(text)
