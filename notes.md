= ToDO =

SDFRenderer design :

The renderer has input :
[1] VoxelEntry[] (int3 pos/ int ptr/ int offset)
[2] Voxel[] - voxel bricks

The renderer works in 3 passes :
[1]  Pass 1 : Draw with front-face culling
	* Uses VoxelEntry as input, draws Brick sized cubes around VoxelEntry.pos
	* Draw depth to fbo_back's depth texture, and output VoxelEntry.ptr into integer texture

[2] Pass 2 : Draw with back-face culling
	* Uses VoxelEntry as input, draws Brick sized cubes around VoxelEntry.pos
	* Draw depth to fbo_front's depth texture, and output VoxelEntry.ptr into integer texture
