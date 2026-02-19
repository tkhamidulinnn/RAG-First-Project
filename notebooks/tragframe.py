"""
Ultra-minimal mentor-facing framework.

Usage:
    import tragframe
    vd = tragframe.Vectordatabase()
    vd.Update("data")
    rag = tragframe.Rag(vd)
    output = rag.retrieve("What is RAG?")
    print(output)
"""

from source.week3_rag_skeleton import Rag, VectorDatabase


class Vectordatabase(VectorDatabase):
    pass
