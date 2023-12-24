# Performance Comparison of Tracking Methods

For evaluation use the TrackEval framework: [JonathonLuiten/TrackEval](https://github.com/JonathonLuiten/TrackEval).

Also, version with the updated numpy package can be found here: [Robotmurlock/TrackEval](https://github.com/Robotmurlock/TrackEval)

## DanceTrack validation dataset

All configs can be found in `configs` repository directory. 
Default ReID model is the fast-reid SBS(S50).

| Method Name     | Description                                            | HOTA | MOTA | IDF1 | config            |
|-----------------|--------------------------------------------------------|------|------|------|-------------------|
| FastTracker     | no motion filter + IoU + Greedy                        | 46.0 | 88.7 | 44.4 | fast.yaml         |
| Bot-SORT        | [arxiv: Bot-SORT](https://arxiv.org/abs/2206.14651)    | 51.3 | 90.4 | 52.2 | botsort.yaml      |
| SORT            | [arxiv: SORT](https://arxiv.org/pdf/1602.00763.pdf)    | 51.4 | 89.6 | 51.2 | sort.yaml         |
| ByteTrack       | [arxiv: ByteTrack](https://arxiv.org/abs/2110.06864)   | 52.3 | 90.5 | 53.3 | bytetrack.yaml    |
| SORT-ReID       | SORT + ReID                                            | 52.3 | 89.8 | 52.1 | sort_reid.yaml    |
| SparseTrack     | [arxiv: SparseTrack](https://arxiv.org/abs/2306.05238) | 52.4 | 90.0 | 52.7 | sparsetrack.yaml  |

Custom architectures:

| Method Name    | Description                                             | HOTA | MOTA | IDF1 | config             |
|----------------|---------------------------------------------------------|------|------|------|--------------------|
| MoveSORT       | SORT + Move association                                 | 52.7 | 89.9 | 53.0 | movesort.yaml      |
| MoveSORT + CMC | SORT + Move association + CMC                           | 53.1 | 90.1 | 53.3 | movesort_gmc.yaml  |
| MoveByte       | Byte + Move association                                 | 53.8 | 90.5 | 54.7 | movebyte.yaml      |
| MoveByte + CKF | Byte + Move association + Confidence KF and association | 56.0 | 90.5 | 57.7 | movebyte_conf.yaml |
