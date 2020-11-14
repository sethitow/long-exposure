![Attempted long exposure](results/every_frame.png)

An attempt to create a long exposure of the SpaceX launch video I took at Vandenberg AFB.

## Performance
With Numba, performance is usable. 

| Number of Frames | Processing Time (s) |
|------------------|---------------------|
| 52               | 22.819455109000003  |
| 103              | 46.917218029        |
| 307              | 127.096594074       |
| 1533             | 746.9913527489999   |

Without Numba, processing 52 frames took over 25 minutes. 