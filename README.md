### Segmented Style Transfer

<img src="./payload/test.jpg" height="200px"></img>
<img src="./payload/test_MASK.png" height="200px"></img>
<img src="./payload/test_FST.png" height="200px"></img>
<img src="./payload/test_MASK+FST.png" height="200px"></img>

Selectively apply neural style transfer upon specific semantic segments within an image. Facilitated with instance segmentation and fast style transfer. Execute with `python segmented_style_transfer.py`. Pass `segment` index as argument into `PartialStyleTransfer`. 