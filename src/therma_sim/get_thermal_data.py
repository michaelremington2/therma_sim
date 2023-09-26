#!/usr/bin/python
import fiona
import rasterio
import rasterio.mask
import matplotlib.pyplot as plt
from rasterio.plot import show


class GetThermalData(object):
    def __init__(self):
        pass

    def get_landscape_shape(self, landscape_raster_fp):
        '''Enter the shape file file path to be read with fiona'''
        with fiona.open(landscape_raster_fp, "r") as shapefile:
            landscape_shp = [feature["geometry"] for feature in shapefile]
        return landscape_shp

    def extract_environmental_data_shape(self, env_data_tiff_fp, landscape_shp):
        with rasterio.open(env_data_tiff_fp) as src:
            out_image, out_transform = rasterio.mask.mask(src, landscape_shp, crop=True)
            out_meta = src.meta
            out_meta.update({"driver": "GTiff",
                             "height": out_image.shape[1],
                             "width": out_image.shape[2],
                             "transform": out_transform})
        return out_image, out_meta

    def write_output_env_tiff(self, fp, out_image, out_meta):
        with rasterio.open(fp, "w", **out_meta) as dest:
            dest.write(out_image)

    def main(self, landscape_shp_fp, env_data_tiff_fp, output_fp):
        ls_shp = self.get_landscape_shape(landscape_shp_fp)
        output_therma_raster, meta_data = env_data_shp.extract_environmental_data_shape(env_data_tiff_fp = env_data_tiff_fp,
                                                                                        landscape_shp = ls_shp)
        env_data_shp.write_output_env_tiff(fp = output_fp,
                                           out_image = output_therma_raster,
                                           out_meta = meta_data)


if __name__ ==  "__main__":
    env_data_shp = GetThermalData()
    ls_shp = env_data_shp.get_landscape_shape('/home/mremington/Documents/therma_sim/Raster-Layers/landscape-polygons/canada/Canada.shp')
    output_therma_raster, meta_data = env_data_shp.extract_environmental_data_shape(env_data_tiff_fp = '/home/mremington/Documents/therma_sim/Raster-Layers/bioclim_data/wc2.1_30s_bio_1.tif',
                                                                                    landscape_shp = ls_shp)
    env_data_shp.write_output_env_tiff(fp = '/home/mremington/Documents/therma_sim/Raster-Layers/landscape-polygons/canada/CAN_mean_temp.tif',
                                       out_image = output_therma_raster,
                                       out_meta = meta_data)
    src = rasterio.open('/home/mremington/Documents/therma_sim/Raster-Layers/landscape-polygons/canada/CAN_mean_temp.tif')
    plt.imshow(src.read(1), cmap='hot')
    plt.show()
