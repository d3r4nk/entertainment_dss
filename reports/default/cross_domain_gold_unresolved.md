# Unresolved cross-domain gold-pair ideas

These plausible candidates were not added because the corresponding exact title
was not present in the default MovieLens/Steam catalogues, or the relation could
not be justified from the catalogue names alone:

| Candidate | Why unresolved |
|---|---|
| Blade Runner ↔ a Blade Runner-titled game | No Blade Runner game title appears in the Steam catalogue; the included Cyberpunk 2077 pair is explicitly thematic instead. |
| The Matrix ↔ Enter the Matrix | No Enter the Matrix title appears in the Steam catalogue; the included Deus Ex pair is thematic, not a franchise claim. |
| Jaws ↔ Jaws Unleashed | No Jaws-titled game appears in the Steam catalogue; Man of Medan is marked thematic only. |
| Sonic the Hedgehog (film) ↔ Sonic game | No Sonic feature film was found in the default MovieLens catalogue. |
| The Road ↔ The Last of Us | No matching The Last of Us game or film title appears in the default Steam/MovieLens catalogues. |
| Ghost of Tsushima ↔ The Last Samurai | Ghost of Tsushima was not found in the default Steam catalogue; Sekiro is used as a separately justified thematic match instead. |

Candidates were checked against `movies.parquet` and `games.parquet`; fuzzy
matches below a defensible title/relation threshold were not promoted to gold.
