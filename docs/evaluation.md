# Performance Comparison of Tracking Methods

For evaluation use the TrackEval framework: [JonathonLuiten/TrackEval](https://github.com/JonathonLuiten/TrackEval).

Also, version with the updated numpy package can be found here: [Robotmurlock/TrackEval](https://github.com/Robotmurlock/TrackEval)

## DanceTrack validation dataset

All configs can be found in `configs` repository directory.

| Method Name    | Description                                            | HOTA | MOTA | IDF1 | config            |
|----------------|--------------------------------------------------------|------|------|------|-------------------|
| FastTracker    | no motion filter + IoU + Greedy                        | 46.0 | 88.7 | 44.4 | fast.yaml         |
| SORT           | [arxiv: SORT](https://arxiv.org/pdf/1602.00763.pdf)    | 51.4 | 89.6 | 51.2 | sort.yaml         |
| Bot-SORT       | [arxiv: Bot-SORT](https://arxiv.org/abs/2206.14651)    | 51.8 | 90.3 | 52.4 | botsort.yaml      |
| ByteTrack      | [arxiv: ByteTrack](https://arxiv.org/abs/2110.06864)   | 52.0 | 90.4 | 52.4 | bytetrack.yaml    |
| MoveSORT       | SORT + Move association                                | 52.7 | 90.1 | 52.9 | movesort.yaml     |
| SparseTrack    | [arxiv: SparseTrack](https://arxiv.org/abs/2306.05238) | 53.0 | 90.0 | 53.2 | sparsetrack.yaml  |
| MoveSORT + CMC | SORT + Move association + CMC                          | 53.1 | 90.1 | 53.3 | movesort_gmc.yaml |
