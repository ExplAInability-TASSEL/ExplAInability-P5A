# Developping a web application

### [*Master Github depository*](https://github.com/ExplAInability-TASSEL/ExplAInability-P5A)
-----

This **README** is meant to explain how the **preprocessing** works and how to use it.

# **Requirements**
First thing you'll need is to **install** the necessary **requirements**.  
We recommand to use a **virtual environment** to avoid any conflict with your current **python installation**.  
You can use [**conda**](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to create a virtual environment and install the requirements in it as follow:
```bash
conda create -n myenv python=3.8
conda activate myenv
pip install -r requirements.txt
``` 

# **Overview**
The input files are not present in this repository, as they are too heavy.  
The ouput files are not there either, you'll have to run the code to generate them.

The **preprocessing** is divided in **3 steps**:
1. Extract the **segments** that are **inside** the **polygons** (included from 0 to 100% of their area).
2. **Normalize** the **data** per **band**.
3. **Compute** the **component-level stats** (in our case mean) for each of the 10 bands, giving a total data size (73 x 10) -> 73 dates x 1 descriptor x 10 bands. This will be your **input data** for the **classifier**.

# **Approach**
In our analysis, we consider three different spatial entities: **1) pixels**, **2) polygons**, and **3) segments**.

The data we provide, which includes **raster data**, is specified at the **pixel level**. This means that each individual pixel in our images carries specific data.

However, the **ground truth data** was generated at the **polygon level**. Each polygon, which represents a contiguous area in the image and therefore consists of multiple pixels, has been assigned a specific label.

In our proposed approach, we work at the **segment level**. Segments are different from pixels and polygons; they are created using an unsupervised segmentation approach called **SLIC**. Typically, a segment (comprising around **50 pixels**) is smaller than a polygon. This is why you'll need to select the segments that are contained within each polygon.

# **How to use**
You can run the **preprocessing** by running the **`process_data.py`** file directly.  
```bash
python process_data.py
```
That will ask you to enter the **path** to the **input files** and the **path** to the **output files**. Then it will run the preprocessing and save the output files in the specified output path:
```bash
python process_data.py
Enter the path to the data folder: path/to/input
Enter the path to the output folder: path/to/output
Processing data...
```

You can also go through the different **notebooks** to see how the preprocessing works and how to use it:
- `1_data_analyze.ipynb` analyze the input data.
- `2_extract_included_segments.ipynb` extract segments partially (0%->100%) included in polygons for training set.
- `2_extract_segments.ipynb` extract all segments and their pixels for inference set.
- `3_segment_analyze.ipynb` analyze the segment extraction results.
- `4_normalization.ipynb` normalize the acquisition data.
- `5_730_conversion.ipynb` convert pixel coordinates for each segment (obtained with `2_extract_segments.ipynb`) to their 730 values vector (73 acquisitions for 10 bands --> 730 values).

# **Data description**
**Study area**: **2000 km^2** in Burkina Faso around the town of **Koumbia**.  
**Coordinates**: **`(11.37, -3.89)`** top left and **`(10.96, -3.42)`** bottom left corner in **(latitude, longitude)**.  
**Satellite data**: from **Sentinel 2** mission, with **73 acquired images** from the study area over the year of 2020 (starting from **January 5th** and a new acquisition **every 5 days** until **December 30th**), with a **10m/px spatial resolution**.
- **Satellite raster data (Sentinel 2)**
    - **10 spectral bands**, one per file (**`s2_2020_B02.tif`**, **`s2_2020_B03.tif`**, etc...)
    - Each file **`s2_2020_BXX.tif`**: contains an array of size **`(73, 4513, 5183)`**=(nb. dates, height, width).
- **`Class_ID_2020.tif`**
    - array of size **`(1, 4513, 5183)`** containing the class ID (1 to 8) on their corresponding location and NaN entries for non-annotated pixels.
    - There is a total of **79962 annotated pixels**, grouped in polygons where all pixels within a same polygon have the same label.
    - Classes 1 to 8 correspond respectively to ['Cereals', 'Cotton', 'Oleag./Legum.', 'Grassland', 'Shrubland', 'Forest', 'Baresoil', 'Water']
- **`Polygon_ID.tif`**
    - array of size **`(1, 4513, 5183)`** containing the polygon ID (0 to 997) on their corresponding location and NaN entries for non-annotated pixels.
- **`Segment_ID.tif`**
    - array of size **`(1, 4513, 5183)`** containing the segment ID (from 1 to 467957). Each segment contains on average **50 pixels**.


# **Architecture**

The following architecture is used to store the data:
```bash
data
│   ├───contained_segments
│   ├───contained_segments_ordered
│   ├───extraction_train
│   ├───min_max_outputs
│   └───tif_file
```
Where:
- **`contained_segments`**: contains the **segments** that are **inside** the **polygons** (included from 0 to 100% of their area).
- **`contained_segments_ordered`**: contains the **segments** that are **inside** the **polygons** (included from 0 to 100% of their area) **ordered** for perimeter to work in webmap.
- **`extraction_train`**: contains the train **segments**, hence already supervised (no need to be inside polygons).
- **`min_max_outputs`**: contains the **min** and **max** values for each band (useful for normalization per band).
- **tif_file**: contains the **raster data** (Sentinel 2) of one band for one acquisition date.

The useful_notebooks folder contains the notebooks used to extract poart of the map, extract training data or test new features, they are not essential in the processing pipeline.

# Normalization insight
1) Normalize data **per band**
	- For each band **separately**, normalize data to the interval **[0, 1]**:
		```bash 
		norm_data = np.clip( (data - min_val) / (max_val - min_val), 0, 1)
		```
	    Where **min_val** and **max_val** correspond respectively to the **2%** and **98%** percentile on the data. Any values **outside** the range [min_val, max_val] (eventual outliers) are **clipped to 0 or 1**.
2) Obtain **ground truth labels** per segment (originally grouped per polygons):
	- Select **segments** completely **contained inside a polygon** and attribute them the corresponding **polygon label**.
3) For each **segment**:
	- Compute **components** (2 components for each segment computed by **k-means clustering**)
	- Compute **component-level stats** (median , mean, std) for each of the 10 bands, giving a total data size (73 x 30) -> 73 dates x 3 descriptors times 10 bands. This will be your **input data** for the **classifier**.



