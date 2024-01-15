import os
import numpy as np
import rasterio
import rasterio.features
import pandas as pd
import rioxarray as rxr
import time

def open_tif(file, infos=False):
    """
    Open a GeoTIFF file using rioxarray.

    Parameters:
    - file (str): The path to the GeoTIFF file.
    - infos (bool): If True, print information about the raster.

    Returns:
    - o_file (rioxarray.Dataset): The opened GeoTIFF file.
    """
    with rxr.open_rasterio(file) as ds:
        if infos:
            print(ds)
            print(ds.rio.crs)
            print(ds.rio.bounds())
            print(ds.rio.resolution())
            print(ds.rio.shape)
            print(ds.rio.transform())
            print(ds.rio.count)
    o_file = rxr.open_rasterio(file)
    return o_file

def find_contained_segments(class_ID, segment_ID, polygon_ID, output_file, included_pixel_percentage=100):
    """
    Find contained segments based on given conditions and write to a file.

    Parameters:
    - class_ID (numpy.ndarray): Array containing class data.
    - segment_ID (numpy.ndarray): Array containing segment data.
    - polygon_ID (numpy.ndarray): Array containing polygon data.
    - output_file (File): The file to write the results.
    - included_pixel_percentage (float): Percentage of included pixels (default is 100).
    """
    polygons = np.unique(polygon_ID)
    polygons = polygons[~np.isnan(polygons)]
    iterations = len(polygons)

    for i, polygon in enumerate(polygons):
        print(f"Iteration {i + 1}/{iterations}")
        polygon_pixels = np.where(polygon_ID == polygon)
        polygon_pixels = np.array(polygon_pixels).T

        polygon_segments = []
        for pixel in polygon_pixels:
            polygon_segments.append(segment_ID[pixel[0], pixel[1]])
        polygon_segments = np.unique(polygon_segments)

        for segment in polygon_segments:
            segment_pixels = np.where(segment_ID == segment)
            segment_pixels = np.array(segment_pixels).T
            total_pixels = len(segment_pixels)

            segment_pixel_polygon = np.where((segment_ID == segment) & (polygon_ID == polygon))
            segment_pixel_polygon = np.array(segment_pixel_polygon).T
            included_pixels = len(segment_pixel_polygon)

            percentage_included = (included_pixels / total_pixels) * 100

            if percentage_included >= included_pixel_percentage:
                class_id = class_ID[segment_pixel_polygon[0, 0], segment_pixel_polygon[0, 1]]
                pixel_coords = [list(pixel) for pixel in segment_pixels]
                output_file.write(f"Segment ID: {segment}, Polygon ID: {polygon}, Class ID: {class_id}, Pixels: {pixel_coords}\n")

def retrieve_all_segments(segment_ID, output_file):
    """
    Retrieve all segments and write to a file.

    Parameters:
    - segment_ID (numpy.ndarray): Array containing segment data.
    - output_file (File): The file to write the results.
    """
    _, indices = np.unique(segment_ID, return_inverse=True)
    unique_segments, segment_counts = np.unique(indices, return_counts=True)
    total_pixels = len(segment_ID.flatten())

    for i, count in zip(unique_segments, segment_counts):
        segment_indices = np.argwhere(indices == i).flatten()
        pixel_coords = [list(np.unravel_index(idx, segment_ID.shape)) for idx in segment_indices]
        output_file.write(f"Segment ID: {i}, Polygon ID: 0, Class ID: 0, Pixels: {pixel_coords}\n")

        print(f"Processed {sum(segment_counts[:i+1])}/{total_pixels} pixels", end='\r')

    print("Processing complete!")

def normalize_and_save_band(band, output_folder):
    """
    Normalize band data and save to a GeoTIFF file.

    Parameters:
    - band (str): Path to the band GeoTIFF file.
    - output_folder (str): Path to the output folder.
    """
    with rasterio.open(band) as src:
        data_array = src.read()
        min_val = np.percentile(data_array, 2)
        max_val = np.percentile(data_array, 98)
        norm_band = np.clip((data_array - min_val) / (max_val - min_val), 0, 1).astype('float32')

        print(f"Min: {min_val}, Max: {max_val}, Mean: {np.mean(norm_band)}")

        output_path = os.path.join(output_folder, f'normalized_{os.path.basename(band)}.tif')
        profile = src.profile.copy()
        profile.update(count=1, dtype='float32')

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(norm_band)

def load_data_from_file(contained_file):
    """
    Load data from a file and create a dataframe.

    Parameters:
    - contained_file (str): Path to the file containing data.

    Returns:
    - df (pd.DataFrame): The created dataframe.
    """
    segment_id = []
    polygon_id = []
    class_id = []
    pixels = []
    pixels_per = []

    with open(contained_file) as f:
        next(f)  # Skip the header line
        for line in f:
            data = line.strip().split(',')
            segment_id.append(data[0])
            polygon_id.append(data[1])
            class_id.append(data[2])

            pixel_str = line.split('"')[1]
            pixel_str = pixel_str.replace('), (', '|').strip('][').replace('(', '').replace(')', '')
            pixel_list = [list(map(int, point.split(', '))) for point in pixel_str.split('|')]
            pixels.append(pixel_list)

            pixel_per_str = line.split('"')[3]
            pixel_per_str = pixel_per_str.replace('), (', '|').strip('][').replace('(', '').replace(')', '')
            pixel_per_list = [list(map(int, point.split(', '))) for point in pixel_per_str.split('|')]
            pixels_per.append(pixel_per_list)

    df = pd.DataFrame()
    df['segment_id'] = segment_id
    df['polygon_id'] = polygon_id
    df['class_id'] = class_id
    df['pixels'] = pixels
    df['Perimeter Pixels'] = pixels_per

    return df

def process_data(folder, output_folder):
    """
    Process data including finding contained segments, retrieving all segments, and normalizing data.

    Parameters:
    - folder (str): Path to the data folder.
    - output_folder (str): Path to the output folder.
    """
    os.makedirs(output_folder, exist_ok=True)

    class_ID = os.path.join(folder, "class_ID_2020.tif")
    segment_ID = os.path.join(folder, "segment_ID_2020.tif")
    polygon_ID = os.path.join(folder, "polygon_ID.tif")

    class_ID_array = open_tif(class_ID)
    segment_ID_array = open_tif(segment_ID)
    polygon_ID_array = open_tif(polygon_ID)

    # Output files
    output_file_contained = os.path.join(output_folder, "contained_segments.txt")
    output_file_all_segments = os.path.join(output_folder, "contained_all_segments.txt")
    output_file_normalized = os.path.join(output_folder, "normalized_data.xlsx")

    # Find contained segments
    with open(output_file_contained, 'w') as output_file:
        find_contained_segments(class_ID_array.data[0], segment_ID_array.data[0], polygon_ID_array.data[0],
                                output_file, included_pixel_percentage=50)

    # Retrieve all segments
    with open(output_file_all_segments, 'w') as output_file:
        retrieve_all_segments(segment_ID_array.data[0], output_file)

    # Normalize data
    output_band_folder = os.path.join(output_folder, 'normalized_gathered_band_final_output')
    os.makedirs(output_band_folder, exist_ok=True)

    min_max_folder = os.path.join(output_folder, 'min_max_outputs')
    os.makedirs(min_max_folder, exist_ok=True)

    list_bands = ['02', '03', '04', '05', '06', '07', '08', '8A', '11', '12']

    for band in list_bands:
        band_file = os.path.join(folder, f's2_2020_B{band}.tif')
        normalize_and_save_band(band_file, output_band_folder)

    # Process segment data
    contained_file_50 = os.path.join(output_folder, 'contained_segments.txt')
    dataframe_segment = load_data_from_file(contained_file_50)

    # Create output dataframe
    dataframe_segment_output = dataframe_segment.copy()
    dataframe_segment_output['pixels_value'] = dataframe_segment_output['pixels'].apply(lambda x: x[:])

    for index_output, row_output in dataframe_segment_output.iterrows():
        for i_output, pixel_output in enumerate(row_output['pixels_value']):
            row_output['pixels_value'][i_output] = []

    start = 1
    acquisition = 73

    for band in list_bands:
        file_path = os.path.join(output_band_folder, f'normalized_s2_2020_B{band}.tif')
        with rasterio.open(file_path) as src:
            for period in range(start, acquisition + 1):
                band_acquisition = src.read(period)
                for index, row in dataframe_segment.iterrows():
                    for i, pixel in enumerate(row['pixels']):
                        x, y = pixel
                        value = band_acquisition[x, y]
                        dataframe_segment_output.at[index, 'pixels_value'][i].append(value)

    # Save output dataframe as Excel file
    dataframe_segment_output.to_excel(output_file_normalized, index=False)

    print("Process completed!")

if __name__ == "__main__":
    input_folder = input("Enter the path to the data folder: ")
    output_folder = input("Enter the path to the output folder: ")

    start_time = time.time()
    print("Processing data...")
    process_data(input_folder, output_folder)
    print(f"Time taken: {time.time() - start_time} seconds")
