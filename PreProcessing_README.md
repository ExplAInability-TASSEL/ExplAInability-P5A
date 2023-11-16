------------------
1) Normalize data per band
	- For each band separately, normalize data to the interval [0, 1]:
	    norm_data = np.clip( (data - min_val) / (max_val - min_val), 0, 1)
	    where min_val and max_val correspond respectively to the 2% and 98% percentile on the data. Any values outside the range [min_val, max_val] (eventual outliers) are clipped to 0 or 1.
2) Obtain ground truth labels per segment (originally grouped per polygons):
	- Select segments completely contained inside a polygon and attribute them the corresponding polygon label.
3) For each segment:
	- Compute components (2 components for each segment computed by k-means clustering)
	- Compute component-level stats (median , mean, std) for each of the 10 bands, giving a total data size (73 x 30) -> 73 dates x 3 descriptors times 10 bands. This will be your input data for the classifier.


Note that there are three different spatial entities on the analysis: 1) pixel, 2) polygon, 3) segment.
All the data we provide (raster data) are specified at the pixel level.
However, the ground truth data was generated at the polygon level. To each polygon (which corresponds to a contiguous area on the image and therefore corresponds to a certain number of pixels) a label was attributed.
Finally, in the proposed approach we work at the segment level, which is a different spatial entity. The segments were obtained by an unsupervised segmentation approach called SLIC. The segments (around 50 pixels) are usually of smaller size than the polygons. That's why you will need to select the segments that lie inside each polygon.





===============================
DATA DESCRIPTION
----------------
Study area: 2000 km^2 in Burkina Faso around the town of Koumbia (see Figure 1 below).
Coordinates: (11.37, -3.89) top left and (10.96, -3.42) bottom left corner in (latitude, longitude)
Satellite data: from Sentinel 2 mission, with 73 acquired images from the study area over the year of 2020 (starting from January 5th and a new acquisition every 5 days until December 30th), with a 10m/px spatial resolution.
- Satellite raster data (Sentinel 2)
	- 10 spectral bands, one per file (s2_2020_B02.tif, s2_2020_B03.tif, etc...)
	- Each file s2_2020_BXX.tif: contains an array of size (73, 4513, 5183)=(nb. dates, height, width).
- Class_ID_2020.tif
	- array of size (1, 4513, 5183) containing the class ID (1 to 8) on their corresponding location and NaN entries for non-annotated pixels.
	- There is a total of 79962 annotated pixels, grouped in polygons where all pixels within a same polygon have the same label.
	- Classes 1 to 8 correspond respectively to ['Cereals', 'Cotton', 'Oleag./Legum.', 'Grassland', 'Shrubland', 'Forest', 'Baresoil', 'Water']
- Polygon_ID.tif (seex Figure 1)
	- array of size (1, 4513, 5183) containing the polygon ID (0 to 997) on their corresponding location and NaN entries for non-annotated pixels.
- Segment_ID.tif (see Figure 2)
	- array of size (1, 4513, 5183) containing the segment ID (from 1 to 467957). Each segment contains on average 50 pixels.
Opening .tif file (use rasterio library):
	import rasterio as rio
	src = rio.open("filename.tif")
	array = src.read()