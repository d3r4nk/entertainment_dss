"""Recommender models over the processed Parquet artefacts.

Models (PROJECT_SPEC.md section 7):

* :mod:`popularity`  - most interacted items
* :mod:`itemknn`     - cosine on the sparse item x user matrix
* :mod:`mf`          - implicit ALS (falls back to TruncatedSVD)
* :mod:`content`     - cosine against the mean embedding of the user's likes
* :mod:`kg_scoring`  - theme/genre profile overlap
* :mod:`hybrid`      - weighted blend of min-max normalised scores

Every model shares the :class:`~entdec.recsys.common.InteractionsView`
train view so evaluation code can treat them uniformly.
"""
