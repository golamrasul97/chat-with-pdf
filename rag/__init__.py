"""The RAG pipeline, one module per phase of the loop.

    ingest.py  ->  ingest + chunk      (PDF -> overlapping, page-tagged passages)
    index.py   ->  embed + store + retrieve  (passages -> FAISS -> top-k matches)
    chat.py    ->  augment + generate  (matches + question -> cited answer)

config.py holds the settings all three read from the environment.
"""
