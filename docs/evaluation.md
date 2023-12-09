# Performance Comparison of Tracking Methods

For evaluation use the TrackEval framework: [JonathonLuiten/TrackEval](https://github.com/JonathonLuiten/TrackEval).

Also, version with the updated numpy package can be found here: [Robotmurlock/TrackEval](https://github.com/Robotmurlock/TrackEval)

## DanceTrack validation dataset

| Method Name    | Description                                     | HOTA | MOTA | IDF1 |
|----------------|-------------------------------------------------|------|------|------|
| MoveSort       | SORT + Move association                         | 52.7 | 90.1 | 52.9 |
| MoveSort + CMC | SORT + Move association + GMC CMC               | 53.1 | 90.1 | 53.3 |
| ByteTrack      | [ByteTrack](https://arxiv.org/abs/2110.06864)   | ---- | ---- | ---- |
| Bot-SORT       | [Bot-SORT](https://arxiv.org/abs/2206.14651)    | ---- | ---- | ---- |
| SparseTrack    | [SparseTrack](https://arxiv.org/abs/2306.05238) | ---- | ---- | ---- |
