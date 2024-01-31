# Performance Comparison of Tracking Methods

For evaluation use the TrackEval framework: [JonathonLuiten/TrackEval](https://github.com/JonathonLuiten/TrackEval).

Also, version with the updated numpy package can be found here: [Robotmurlock/TrackEval](https://github.com/Robotmurlock/TrackEval)

## DanceTrack validation dataset

All configs can be found in `configs/dancetrack` repository directory. 
Default ReID model is the fast-reid SBS(S50). Default detection model is the YOLOX model.

| Method Name | Description                                            | HOTA | MOTA | IDF1 | config           |
|-------------|--------------------------------------------------------|------|------|------|------------------|
| FastTracker | no motion filter + IoU + Greedy                        | 46.0 | 88.7 | 44.4 | fast.yaml        |
| Bot-SORT    | [arxiv: Bot-SORT](https://arxiv.org/abs/2206.14651)    | 51.3 | 90.4 | 52.2 | botsort.yaml     |
| SORT        | [arxiv: SORT](https://arxiv.org/pdf/1602.00763.pdf)    | 51.5 | 89.6 | 51.2 | sort.yaml        |
| ByteTrack   | [arxiv: ByteTrack](https://arxiv.org/abs/2110.06864)   | 52.9 | 90.8 | 54.4 | bytetrack.yaml   |
| SparseTrack | [arxiv: SparseTrack](https://arxiv.org/abs/2306.05238) | 52.4 | 90.0 | 52.7 | sparsetrack.yaml |
| SORT-ReID   | SORT + ReID (FastReID SBS-S50)                         | 56.3 | 89.9 | 56.9 | sort_reid.yaml   |

Custom architectures:

| Method Name    | Description                            | HOTA | MOTA | IDF1 | config             |
|----------------|----------------------------------------|------|------|------|--------------------|
| MoveSORT       | SORT + Move                            | 52.7 | 89.9 | 53.0 | movesort.yaml      |
| MoveSORT + CMC | SORT + Move + CMC                      | 53.1 | 90.1 | 53.3 | movesort_gmc.yaml  |
| MoveByte       | Byte + Move                            | 53.8 | 90.5 | 54.7 | movebyte.yaml      |
| MoveByte + CKF | Byte + Move + Conf                     | 56.0 | 90.5 | 57.7 | movebyte_conf.yaml |
| DeepMoveSORT   | Byte + ReID + HVC + Conf + TransFilter | 60.8 | 90.9 | 65.0 | coming_soon.yaml   |

## MOT17 validation dataset

All configs can be found in `configs/mot17` repository directory. 
Default ReID model is the fast-reid SBS(S50). Default detection model is the YOLOX model.

| Method Name   | Description                                          | HOTA | MOTA | IDF1 | config            |
|---------------|------------------------------------------------------|------|------|------|-------------------|
| SORT          | [arxiv: SORT](https://arxiv.org/pdf/1602.00763.pdf)  | 68.0 | 78.7 | 79.4 | sort.yaml         |
| ByteTrack     | [arxiv: ByteTrack](https://arxiv.org/abs/2110.06864) | 68.2 | 78.6 | 80.1 | bytetrack.yaml    |
| Bot-SORT      | [arxiv: Bot-SORT](https://arxiv.org/abs/2206.14651)  | 69.4 | 79.6 | 82.2 | botsort.yaml      |
| Bot-SORT-ReID | Bot-SORT + ReID (FastReID SBS-S50)                   | 70.0 | 79.7 | 82.7 | botsort_reid.yaml |