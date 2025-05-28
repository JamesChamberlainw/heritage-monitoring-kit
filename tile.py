import ee
import pandas as pd 

ee.Authenticate()
ee.Initialize(project="jameswilliamchamberlain")

class tile():

    tiles = None

    def __init__(self, ee):
        self.ee = ee

    def name_tile(f, str_start="tile"):
        coords = ee.List(f.geometry().bounds().coordinates().get(0))

        ll = ee.List(coords.get(0))  # lower-left
        ur = ee.List(coords.get(2))  # upper-right

        lon_min = ee.Number(ll.get(0)).multiply(1e6).round().toInt() # .round().multiply(1e6).toInt()
        lat_min = ee.Number(ll.get(1)).multiply(1e6).round().toInt() # .round().multiply(1e6).toInt()
        lon_max = ee.Number(ur.get(0)).multiply(1e6).round().toInt() # .round().multiply(1e6).toInt()
        lat_max = ee.Number(ur.get(1)).multiply(1e6).round().toInt() # .round().multiply(1e6).toInt()

        file_name = ee.String(f'{str_start}_') \
            .cat(lon_min.format('%06d')).cat('_') \
            .cat(lat_min.format('%06d')).cat('_') \
            .cat(lon_max.format('%06d')).cat('_') \
            .cat(lat_max.format('%06d'))

        return f.set('id', file_name)
    
    def name_chunk(f, str_start="chunk"):
        return name_tile(f, str_start)
    

    def net(sub_region, collection="COPERNICUS/S2", band="B4", tile_size_m=50, name_tile_fn=name_tile):
        """
            Fishnet function, that creates a grid of tiles (50m x 50m) aligned with the projection. 

            Due to limitations with ee.
        
        """
        
        s2 = ee.ImageCollection(collection) \
            .filterBounds(sub_region) \
            .filterDate("2024-01-01", "2024-12-30") \
            .first()

        projection = s2.select(band).projection()
        px_coords = ee.Image.pixelCoordinates(projection)

        tile_px = tile_size_m // 10  # 50/10 = 5px (50mx50m)

        tile_ids = px_coords.select("x").divide(tile_px).floor() \
            .multiply(1e6).add(px_coords.select("y").divide(tile_px).floor()) \
            .toInt()
        
        # 
        fishnet = tile_ids.reduceToVectors(
            geometry=sub_region,
            geometryType='polygon',
            scale=10,
            bestEffort=True,
            maxPixels=1e13,
        )

        fishnet = fishnet.map(name_tile_fn)

        return fishnet
    
    def create_aligned_tiles(region, 
                             projection, 
                             chunk_size_m=500,
                             tile_size_m=50,
                             name_chunk_fn=name_chunk,
                             name_title_fn=name_tile):
        """
            Given a region, projection and chunk size in meters, this function will generate tiles on the given projection. 

            Args:
                region, ee.Geometry
                projection, ee.Projection
                chunk_size_m, int, size of the chunks in meters (default is 50*10, what is 500m)
                tile_size_m, int, size of the tiles in meters (default is 50)
                name_title_fn, function, function to name the tiles (default is name_tile what can be used as an example)
        """
        chunks = net(region, tile_size_m=tile_size_m * 100)

        fishnet_tiles = []
        for i in range(chunks.size().getInfo()):
            chunk = ee.Feature(chunks.toList(1, i).get(0))
            chunk_tiles = net(chunk.geometry(), tile_size_m=tile_size_m)

            # drop edge tiles 
            fishnet_tiles.append(chunk_tiles)

        return fishnet_tiles
    
    def create_tiles(roi, 
                     tile_size_m=50, 
                     collection="COPERNICUS/S2", 
                     band="B4",
                     full_tiles_only=False,
                     name_tile_fn=name_tile):
        """
            Creates a gird of standardised tiles (e.g., 50m x 50m) that are aligned with the projection of the specified band in the given collection. 
        
            Args:
                roi: ee.Geometry
                tile_size_m: int, size of the tiles in meters (default is 50)
                collection: str, Earth Engine image collection (default is "COPERNICUS/S2")
                band: str, band to use for projection (default is "B4", what is 10m resolution in S2)
                full_tiles_only: bool, if True, only returns full tiles that are completely within the geometry (default is False)
                name_tile_fn: function, function to name the tiles (default is name_tile what can be used as an example)
                name_chunk_fn: function, function to name the chunks (default is name_tile what can be used as an example)                
        """
        
        if roi is None:
            raise ValueError("Error: no roi provided")
        
        raise NotImplementedError("This function is not implemented yet")
    
    def save_locally(self, path=""):
        """
            Saves the tiles to a local path; default is the current working directory.

            Args:
                path: str
        """

        if self.tiles is None:
            raise ValueError("Error: no tiles")

        raise NotImplementedError("This function is not implemented yet, you can find the tiles under tile.tiles")

        # for i in range(len(tiles)):
        #     geemap.ee_to_shp(tiles[i], dir.replace(".shp", f"_{i+1}.shp"))


        # global tiles

        # if m.draw_features is None:
        #     raise ValueError("Error: no roi")

        # roi = m.draw_last_feature.geometry()

        # fishnet_list = create_aligned_tiles(roi, tile_size_m=50)
        
        # tiles = fishnet_list
