# Planet stored as 8³ bricks with all-voxel LOD to orbit

The Planet is stored as sparse 8×8×8 bricks. Each brick holds a 512-bit occupancy mask plus palette-compressed 16-bit Material IDs, and uniform bricks cost nothing. On the CPU the bricks hang off a sparse hash index; on the GPU, off a shallow tree. Distant terrain and the orbital view use the same bricks at coarser mip levels: each brick is drawn at the level its distance from the camera says (LOD bands, each twice as far out and twice as coarse as the one inside it), so one Voxel never covers more than about a pixel. There is no heightfield or mesh fallback. A coarse Voxel is solid when at least half its children are solid. It takes the commonest Material among the solid children that are exposed to air (air against a face, even across a brick border), and the commonest among all its solid children only if none is exposed, so that a few Voxels of soil on rock still read as soil from orbit. Edits propagate up every level, so a crater shows from orbit; since exposure reads across brick borders, an Edit also remakes the coarse bricks over the bricks beside the ones it changed. Planet generation makes coarse bricks directly with a matching rule, which agrees with the pyramid on most exposed Voxels but not all: an exposed coarse Voxel takes the Material at the top of the ground within it. Unedited bricks are a cache that can be regenerated from the seed; only bricks touched by Edits are persisted.

We chose bricks over a sparse voxel octree, a 64-tree or a sparse voxel DAG, for three reasons:

- Edits are trivial on bricks, and Edits are the core mechanic.
- Ray stepping is fast.
- The structure is simple enough for AI agents to get right.

A DAG compresses best, but makes Edits hard and is inflated about 6× by Materials. It stays a later optimisation for far levels or disk, adopted only if measured memory demands it. A single all-voxel representation avoids seams and a second renderer, and keeps caves and Edits visible at distance. GPU memory stays bounded because each LOD ring holds roughly the same number of bricks.
