"""Shared contracts. Each domain has its own module; import the public names from here."""

from contracts.ingestion import PRODUCT_SUCCESSOR, Chunk, DocType, Document, Product

__all__ = ["PRODUCT_SUCCESSOR", "Chunk", "DocType", "Document", "Product"]
