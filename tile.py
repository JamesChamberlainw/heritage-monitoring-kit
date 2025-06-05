import ee
import time

import sys

ee.Authenticate()
ee.Initialize(project="jameswilliamchamberlain")

class tile():

    # end result stored here 
    tiles = None 

    # chunk outlines 
    failed_chunks = None # full to partial-chunk polygoes requiring re-tiling
    chunks = None        # chunk polygons

    def __init__(self):
        self.tiles = None 
        self.failed_chunks = None
        self.chunks = None

    @staticmethod
    def version():
        return "0.9.5"

    def name_tile(f, str_start="tile"):
        coords = ee.List(f.geometry().bounds().coordinates().get(0))

        ll = ee.List(coords.get(0))  # lower-left
        ur = ee.List(coords.get(2))  # upper-right

        lon_min = ee.Number(ll.get(0)).multiply(1e6).round().toInt()
        lat_min = ee.Number(ll.get(1)).multiply(1e6).round().toInt()
        lon_max = ee.Number(ur.get(0)).multiply(1e6).round().toInt()
        lat_max = ee.Number(ur.get(1)).multiply(1e6).round().toInt()

        file_name = ee.String(f'{str_start}_') \
            .cat(lon_min.format('%06d')).cat('_') \
            .cat(lat_min.format('%06d')).cat('_') \
            .cat(lon_max.format('%06d')).cat('_') \
            .cat(lat_max.format('%06d'))

        return f.set('file_name', file_name)
    
    def name_chunk(f, str_start="chunk", name_tile_fn=name_tile):
        return name_tile_fn(f, str_start) # basic function does the same thing but with chunk prefix, kept as separete fn to allow for user re-defining

    def net(self, sub_region, collection, band, tile_size_m=50, name_tile_fn=name_tile):
        """
        Fishnet function: creates a grid of tiles (e.g. 50m x 50m) over sub_region using stable CRS, no forced reprojection.
        """

        # def get_utm_crs_from_region(region):
        #     centroid = region.centroid(ee.ErrorMargin(1))
        #     lon = centroid.coordinates().get(0)
        #     zone = ee.Number(lon).add(180).divide(6).floor().add(1)
        #     return ee.String('EPSG:').cat(ee.Number(32600).add(zone).int().format())

        # crs = get_utm_crs_from_region(sub_region)

        # sub_region = sub_region.transform(crs, ee.ErrorMargin(10))

        s2 = ee.ImageCollection(collection) \
            .filterBounds(sub_region) \
            .filterDate("2024-01-01", "2024-12-30") \
            .first() \
            .select(band)

        projection = s2.select(band).projection().atScale(10)

        px_coords = ee.Image.pixelCoordinates(projection)

        tile_px = tile_size_m // 10  # for 50m at 10m res, tile is 5x5 px

        tile_ids = px_coords.select("x").divide(tile_px).floor() \
            .multiply(1e6).add(px_coords.select("y").divide(tile_px).floor()) \
            .toInt()

        fishnet = tile_ids.reduceToVectors(
            geometry=sub_region,
            geometryType='polygon',
            scale=10,
            # crs=crs,
            bestEffort=True,
            maxPixels=1e13,
        )

        fishnet = fishnet.map(name_tile_fn)

        return fishnet
                            
    def create_aligned_tiles(self, 
                             roi, 
                             collection="COPERNICUS/S2",
                             band="B4", 
                             chunk_size_m=250,
                             tile_size_m=50,
                             name_chunk_fn=name_chunk,
                             name_title_fn=name_tile,
                             flag_full_tiles_only=True):
        """
            Given a region, projection and chunk size in meters, this function will generate tiles on the given projection. 

            Args:
                region, ee.Geometry
                collection, str, Earth Engine image collection (default is "COPERNICUS/S2")
                band, str, band to use for projection (B4 is a 10m band in S2)
                chunk_size_m, int, size of the chunks in meters (default is 50*10, what is 500m)
                tile_size_m, int, size of the tiles in meters (default is 50)
                name_title_fn, function, function to name the tiles (default is name_tile what can be used as an example)
        """

        def classify_tile(tile, expected_area, tolerane=0.1):
            area = tile.geometry().area(maxError=1)
            min_area = ee.Number(expected_area).multiply(1 - tolerane)
            max_area = ee.Number(expected_area).multiply(1 + tolerane)

            def tag(status):
                return tile.set('tile_status', status)
            
            return ee.Algorithms.If(
                area.gte(min_area).And(area.lte(max_area)),
                tag("acceptable"),                              # if within range 
                ee.Algorithms.If(
                    area.lt(min_area),
                    tag("partial"),                             # if smaller (so partial tile)
                    tag("failed")                               # else (failed tile so will be larger and requires re-tiling)
                )
            )

        # split region into manageable chunks (else gee will have an issue and fail)
        chunks = self.net(roi, collection, band, chunk_size_m, name_chunk_fn)

        # create a list of failed polygons (should stay empty but there are issues that can occur)
        failed_polygons = ee.FeatureCollection([]) 

        # for each chunk, create a grid of tiles 
        fishnet_tiles = []

        total_chunks = chunks.size().getInfo()

        for i in range(total_chunks):                                                       # range(chunks.size().getInfo()):
            print(f"Processing chunk {i+1} / {chunks.size().getInfo()} ", flush=True)
            # get current chunk 
            chunk = ee.Feature(chunks.toList(1, i).get(0))

            # tile the chunk 
            chunk_tiles = self.net(chunk.geometry(), collection, band, tile_size_m, name_title_fn)

            # # drop current chunk from chunks
            # chunks = chunks.filter(ee.Filter.neq('file_name', chunk.get('file_name')))

            # Tag acceptable tiles 
            chunk_tiles = chunk_tiles.map(lambda f: classify_tile(f, expected_area=tile_size_m**2, tolerane=0.25))

            failed_tiles = chunk_tiles.filter(ee.Filter.eq('tile_status', 'failed'))

            # append failed tiles to chunks
            if failed_tiles.size().getInfo() > 0:
                for tile in failed_tiles.getInfo()['features']:
                    polygon = ee.Feature(tile).geometry()
                    # add chunks failed to chunks_failed list 
                    # TODO: fix this, adding to loop cuases issues, and looping through after all fail so need to find a work around. 
                    failed_polygons = failed_polygons.merge(ee.FeatureCollection([ee.Feature(polygon).set('file_name', name_chunk_fn(ee.Feature(polygon), str_start="failed_tile"))]))
                    print(f"Failed tile: {tile['properties']['file_name']}", flush=True)

            # # drop failed tiles 
            chunk_tiles = chunk_tiles.filter(ee.Filter.neq('tile_status', 'failed')) 

            if flag_full_tiles_only:
                # filter out partial tiles if flag is set
                chunk_tiles = chunk_tiles.filter(ee.Filter.eq('tile_status', 'acceptable'))

            fishnet_tiles.append(chunk_tiles)

        return fishnet_tiles, failed_polygons, chunks
    
    def create_tiles(self,
                     roi, 
                     tile_size_m=50,
                     chunk_size_multi=100,
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
                attempts: int, number of attempts to re-tile failed chunks (default is 1, meaning no re-tiling)
                name_tile_fn: function, function to name the tiles (default is name_tile what can be used as an example)
                name_chunk_fn: function, function to name the chunks (default is name_tile what can be used as an example)       

            Note/TODO: current version attempts has been set to 1, as current version does not support re-tiling.          
        """
       
        if roi is None:
            raise ValueError("Error: no roi provided")

        self.tiles, self.failed_chunks, self.chunks = self.create_aligned_tiles(
            roi=roi, 
            collection=collection,
            band=band,
            chunk_size_m=tile_size_m * chunk_size_multi,  # 50m x 10 = 500m - seems to be a safe option for chunk size - only sometimes failing 
            tile_size_m=tile_size_m,
            name_chunk_fn=name_tile_fn,
            flag_full_tiles_only=full_tiles_only,
            name_title_fn=name_tile_fn
        )

        return self.tiles, self.failed_chunks, self.chunks

    # TODO TEST THIS FUNCTION 
    def export_tiles_to_gee(tiles):
        """
            Exports tiles to GEE asset
        """

        def count_active_tasks():
            """Counts the number of active tasks in GEE"""
            tasks = ee.batch.Task.list()
            active_tasks = [task for task in tasks if task.status()['state'] == 'RUNNING' or task.status()['state'] == 'READY']
            return len(active_tasks)

        # enumerate through each tile and give respective chunk name 
        for i, tile in enumerate(tiles):
            # chunk_name = f"chunk_{i+1}"
            task = ee.batch.Export.table.toAsset(
                collection=tile,
                description=f"chunk_{i+1}",
                assetId=f"projects/jameswilliamchamberlain/assets/chunks/test_chunk_{i+1}"
            )

            task.start()

            active = count_active_tasks()

            while active > 70:
                print(f"Waiting for active tasks to reduce below 70 Active tasks: {active}")
                time.sleep(60)