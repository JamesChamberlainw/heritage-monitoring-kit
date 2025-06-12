import ee
import time
import math
import warnings

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
        return "0.9.9"

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

    @staticmethod
    def get_closest_point(region, point):
        # points as list of coordinates
        points = ee.List(region.coordinates().get(0)) 

        # add distance from each point to the given point
        points_with_distance = points.map(lambda p: ee.Feature(ee.Geometry.Point(p))
                                        .set('coord', p)
                                        .set('dist', ee.Geometry.Point(p).distance(point)))
        
        # sort feature collection by distance 
        fc = ee.FeatureCollection(points_with_distance)
        sorted_fc = fc.sort('dist')

        closest_point = ee.Feature(sorted_fc.first()).get('coord') 

        return ee.List(closest_point)  # return as list of [lon, lat] coordinates

    @staticmethod
    def translate_geometry(geom, dx, dy):
        """
            geometry translation function

            Args:
                geom: ee.Geometry, geometry to translate
                dx: float, change in x (longitude)  
                dy: float, change in y (latitude)
        """
        # Translate the geometry by dx, dy (dx - change in x and dy - change in y)
        translation_coords = ee.List(geom.coordinates().get(0))

        def shift_point(pt):
            pt = ee.List(pt)
            return [ee.Number(pt.get(0)).add(dx), ee.Number(pt.get(1)).add(dy)]

        shifted_coords = translation_coords.map(shift_point)

        # Wrap in outer list to rebuild polygon
        return ee.Geometry.Polygon([shifted_coords])


    # @staticmethod
    # def add_corner_distance(ref_point, corner):
    #     point = ee.Geometry.Point(corner)
    #     dist = point.distance(ref_point)
    #     return ee.Feature(point, {
    #         'dist': dist,
    #         'coord': corner
    #     })


    def vectoriser(self, region, tile_size_m, rotation_deg=45.0, alignemnt_point=None, name_fn=None):

        # Get the top-left corner of the bounding box
        tl_coords = ee.List(region.bounds().coordinates().get(0)).get(3)
        tl = ee.Geometry.Point(tl_coords)

        # set alignment / reference point if not provided as a known point along region 
        alignemnt_point = self.get_closest_point(region, tl) if alignemnt_point is None else alignemnt_point


        # Rotation center 
        centroid = region.centroid(ee.ErrorMargin(1))
        cx = ee.Number(centroid.coordinates().get(0))
        cy = ee.Number(centroid.coordinates().get(1))

        # Rotation Matrix 
        angle_rad = ee.Number(rotation_deg).multiply(math.pi / 180)
        cos_val = angle_rad.cos()
        sin_val = angle_rad.sin()

        rotation_matrix = ee.Array([
            [cos_val, sin_val.multiply(-1)],
            [sin_val, cos_val]
        ])

        coords = ee.List(region.bounds().coordinates().get(0))

        ll = ee.List(coords.get(0))
        ur = ee.List(coords.get(2)) 
        tl = ee.List(coords.get(1)) # bottom right 
        br = ee.List(coords.get(3)) # top left 

        # bounds of the region seperated out
        lon_min = ee.Number(ll.get(0))
        lat_min = ee.Number(ll.get(1))
        lon_max = ee.Number(ur.get(0))
        lat_max = ee.Number(ur.get(1))

        lat = lat_min.add(lat_max).divide(2)
        lat_rad = lat.multiply(math.pi / 180)

        # size of tile in degrees (using approximations using equator values 110574 and 111320)
        # https://support.oxts.com/hc/en-us/articles/115002885125-Level-of-Resolution-of-Longitude-and-Latitude-Measurements
        tile_deg_y = ee.Number(tile_size_m).divide(110574)
        tile_deg_x = ee.Number(tile_size_m).divide(111320).divide(lat_rad.cos()) 

        x_range = lon_max.subtract(lon_min)
        y_range = lat_max.subtract(lat_min)

        num_tiles_x = x_range.divide(tile_deg_x).ceil().toInt()
        num_tiles_y = y_range.divide(tile_deg_y).ceil().toInt()

        def make_tile(i, j):
            i = ee.Number(i)
            j = ee.Number(j)

            x_min = lon_min.add(i.multiply(tile_deg_x))
            y_min = lat_min.add(j.multiply(tile_deg_y))
            x_max = x_min.add(tile_deg_x)
            y_max = y_min.add(tile_deg_y)

            geom = ee.Geometry.Polygon([[
                [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max], [x_min, y_min]
            ]])

            feature = ee.Feature(geom)
            return name_fn(feature) if name_fn else feature

        tiles = ee.List.sequence(0, num_tiles_x.subtract(1)).map(
            lambda i: ee.List.sequence(0, num_tiles_y.subtract(1)).map(
                lambda j: make_tile(i, j)
            )
        ).flatten()

        features = ee.FeatureCollection(tiles)

        def rotate_feature(feature):
            geom = feature.geometry()
            coords = ee.List(geom.coordinates().get(0))  # outer ring only

            def rotate_coord(coord):
                pt = ee.List(coord)
                x = ee.Number(pt.get(0)).subtract(cx)
                y = ee.Number(pt.get(1)).subtract(cy)
                rotated = rotation_matrix.matrixMultiply(ee.Array([[x], [y]]))
                new_x = ee.Number(rotated.get([0, 0])).add(cx)
                new_y = ee.Number(rotated.get([1, 0])).add(cy)
                return ee.List([new_x, new_y])

            rotated_coords = coords.map(rotate_coord)
            rotated_geom = ee.Geometry.Polygon([rotated_coords])

            return feature.setGeometry(rotated_geom)
        
        features = features.map(rotate_feature)

       # Calculate closed rectangle bounds to alignment_coords
        # Convert to point
        ref_point = ee.Geometry.Point(alignemnt_point)

        # add centroids to all features and calculate distance to reference point to find nearest tile 
        features_with_dist = features.map(
            lambda f: f.set('dist', f.geometry().centroid().distance(ref_point))
        )

        # find nearest tile to reference point
        nearest_tile = ee.Feature(features_with_dist.sort('dist').first())
        tile_coords = ee.List(nearest_tile.geometry().coordinates().get(0))

        # find nearest corner and translate the tile to align 
        def add_distance(corner):
            """Calculates the distance from corner point to the reference point for finding nearest corner."""
            point = ee.Geometry.Point(corner)
            dist = point.distance(ref_point)
            return ee.Feature(point, {
                'dist': dist,
                'coord': corner
            })

        corners_fc = ee.FeatureCollection(tile_coords.map(add_distance))
        nearest_corner = ee.List(ee.Feature(corners_fc.sort('dist').first()).get('coord'))

        # Calcualte Change in x and y between the alignment point and the nearest corner
        dx = ee.Number(alignemnt_point.get(0)).subtract(nearest_corner.get(0))
        dy = ee.Number(alignemnt_point.get(1)).subtract(nearest_corner.get(1))

        features = features.map(lambda f: f.setGeometry(self.translate_geometry(f.geometry(), dx, dy)))

        # -=-=-=-     DEBUG POINTS      -=-=-=-
        # features = features.merge(ee.FeatureCollection([
        #     # ee.Feature(ee.Geometry.Point(ur), {'label': 'UR'}), # upper-right
        #     ee.Feature(ee.Geometry.Point(ll), {'label': 'LL'}), # lower-left
        #     ee.Feature(ee.Geometry.Point(tl), {'label': 'TL'}), # top-left ??? 
        #     # ee.Feature(ee.Geometry.Point(br), {'label': 'BR'}), #  ?? ?? bottom-right ??? 
        #     # ee.Feature(ee.Geometry.Point(alignemnt_point), {'label': 'alignment coords'}),
        # ]))
        # -=-=-=-       DEBUG 2           -=-=-=-
        # return ee.FeatureCollection([
        #     # ee.Feature(ee.Geometry.Point(ur), {'label': 'UR'}), # upper-right
        #     # ee.Feature(ee.Geometry.Point(ll), {'label': 'LL'}), # lower-left
        #     # ee.Feature(ee.Geometry.Point(tl), {'label': 'TL'}), # lower right
        #     # ee.Feature(ee.Geometry.Point(br), {'label': 'BR'}), #  ?? ?? bottom-right ??? 
        #     ee.Feature(ee.Geometry.Point(alignemnt_point), {'label': 'alignment coords'}),
        # ])
        # -=-=-=-       DEBUG END         -=-=-=-

        # name the features
        features = features.map(lambda f: name_fn(f) if name_fn else f)

        return features
    
    @staticmethod
    def to_feature(coord):
        coord = ee.List(coord)
        return ee.Feature(None, {
            'x': coord.get(0),  # longitude
            'y': coord.get(1),  # latitude
            'coord': coord
        })

    def extract_angle(self, region=None, p1=None, p2=None):
        """
            Finds the angle between two points within a region for alignment purposes later on. 

            TODO: fix this logic as its too messy and p1, p2 work but does not align properly. 
        """

        # if region is provided 
        sorted_fc = None
        if region is not None:
            coords = ee.List(region.coordinates().get(0))

            # Sort by latitude descending (top first), then by longitude ascending (leftmost of top)
            fc = ee.FeatureCollection(coords.map(self.to_feature))
            sorted_fc = fc.sort('x', True).sort('y', True) #.sort('y', False) 

            # first two points far left and top left 
            p1 = ee.List(ee.Feature(sorted_fc.toList(2).get(0)).get('coord'))
            p2 = ee.List(ee.Feature(sorted_fc.toList(2).get(1)).get('coord'))
        elif p1 is None or p2 is None:
            raise ValueError("tile.py - `extract_angle` - Either 'region' or both 'p1' and 'p2' must be provided.")
        
        dx = ee.Number(p2.get(0)).subtract(p1.get(0))  #  Δx  change in longitude 
        dy = ee.Number(p2.get(1)).subtract(p1.get(1))  #  Δy  change in latitude 

        i = 1
        _dx = dx
        _dy = dy

        # TODO: figure this out (currently works mostly just need to ensure it works 100% of the time)  
        # if region is provided and (some bad logic that needs ot be fixed - should check if one point is direclty above or below the other)  .abs 
        # likley a good option to remove this in the future as circular logic could be used to avoid this issue if region is provided as a polygon - which it is, so all that needs to be checked is one to the left and one to the right in the order (TODO: figure out how to access that)! 
        while (dx.gt(dy) and dx.divide(dy).lt(1.0)) and sorted_fc is not None: # issue detected so need to remove point fron list and recall this fn
            print("DEBUG POINT FAILURE OCCURS DUE TO THIS POINT HERE 1")
            # restore 
            if i > len(sorted_fc.getInfo()['features']):
                dx = _dx
                dy = _dy
                break

            # check next point in list to see if its a better fit
            p2 = ee.List(ee.Feature(sorted_fc.toList(2).get(i+1)).get('coord')) # THRID 
            dx = ee.Number(p2.get(0)).subtract(p1.get(0))
            dy = ee.Number(p2.get(1)).subtract(p1.get(1))
            
            i += 1 

        angle_rad = dy.atan2(dx)  # NOTE: this gives angle from horizontal axis
        angle_deg = angle_rad.multiply(180).divide(math.pi)

        if region is not None:
            # rotate 90 and ensure within 0-180 range
            angle_deg = ee.Number(angle_deg).add(90).mod(180).multiply(-1)
        else:
            # already aligned correclty using correct points that are guanteed to be aligned properly 
            angle_deg = ee.Number(angle_deg)

        # if region is provided, we need to ensure the angle is within 0-180 range
        # angle_deg = ee.Number(angle_deg).add(90).mod(180).multiply(-1)

        return angle_deg # , debug_points

    def net(self, sub_region, collection, band, tile_size_m=50, name_tile_fn=name_tile, geometryType='polygon', vectoriser=None, full_only=False):
        """
            Fishnet function: creates a grid of tiles (e.g. 50m x 50m)

            ignore full_only for now as it is not implemented yet. 
        """

        s2 = ee.ImageCollection(collection) \
            .filterBounds(sub_region) \
            .filterDate("2024-01-01", "2024-12-30") \
            .first() \
            .select(band)

        projection = s2.select(band).projection().atScale(10)

        if vectoriser is not None:
            # calculate angle of image rotation
            # angle = 0.0 
            # ref_point = None
            # if full_only is False:
            #     angle = self.extract_angle(sub_region)
            # else:
            #     coords = ee.List(sub_region.bounds().coordinates().get(0))
            #     ll = ee.Geometry.Point(ee.List(coords.get(0))) # lower left (bounds)
            #     tl = ee.Geometry.Point(ee.List(coords.get(3))) # top left (bounds)
                
            #     p1 = self.get_closest_point(sub_region, ll)  # lower left point
            #     p2 = self.get_closest_point(sub_region, tl)  # top left point

            #     # return p1 p2 as points to view#
            #     angle = self.extract_angle(p1=p1, p2=p2)  # extract angle from two points
                
            #     ref_point = p1  # reference point for vectoriser
            #     # debug code 
            #     # print(f"Anlge of rotation: {angle.getInfo()} degrees")
            #     # return ee.FeatureCollection([ee.Feature(ee.Geometry.Point(p1), {'label': 'p1'}),
            #     #                             ee.Feature(ee.Geometry.Point(p2), {'label': 'p2'})])
            if full_only is True:
                print("WARNING: this feature is not implemented fully for `net` using defualt angle calculations for estimations - these are a bit off.")

            angle = self.extract_angle(sub_region)

            print(f"Anlge of rotation: {angle.getInfo()} degrees")
            # alternative vectoriser function to use in case of failed tiles
            fishnet = vectoriser(sub_region, tile_size_m, rotation_deg=angle)

            return fishnet
        
        # default behavior
        px_coords = ee.Image.pixelCoordinates(projection)

        tile_px = tile_size_m // 10  # for 50m at 10m res, tile is 5x5 px

        tile_ids = px_coords.select("x").divide(tile_px).floor() \
            .multiply(1e6).add(px_coords.select("y").divide(tile_px).floor()) \
            .toInt()

        if vectoriser is not None:
            # alternative vectoriser function to use in case of failed tiles
            fishnet = vectoriser(sub_region, projection, tile_size_m)

            return fishnet

        fishnet = tile_ids.reduceToVectors(
            geometry=sub_region,
            geometryType=geometryType,
            scale=10,
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
                             flag_allow_estimated_tiles=True,
                             flag_full_tiles_only=True,
                             flag_full_chunks_only=False):
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

        # loop var for total chunks
        total_chunks = chunks.size().getInfo()

        # FULL CHUNKS ONLY FIX - 
        # currently this is a workaround for the remapping issue within failed regions - still useful on itsown as a flag so this will note be removed
        if flag_full_chunks_only: 
            # classify chunks based on expected area 
            chunks = chunks.map(lambda f: classify_tile(f, expected_area=chunk_size_m**2, tolerane=0.01)) # 0.01% error tolerance

            # filter out chunks that are not acceptable
            chunks = chunks.filter(ee.Filter.eq('tile_status', 'acceptable'))

            # recalculate total chunks to avoid issues in main chunking loop
            total_chunks_nwe = chunks.size().getInfo()
            print(f"Total chunks after filtering: {total_chunks_nwe}", flush=True)
            print(f"Total chunks before filtering: {total_chunks}", flush=True)
            if total_chunks_nwe == 0:
                raise ValueError("No acceptable chunks found after filtering. Please check your region and chunk size.")
            
            total_chunks = total_chunks_nwe  # update total chunks to new value
                

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

        if flag_allow_estimated_tiles:
            # allow estimated tiles, these may not follow the exact projection but are a pure-estimate 
            # note function is replaceable so if you have a better way to estimate feel free to replace it. 
            for failed_chunk_polygon in failed_polygons.getInfo()['features']:
                failed_chunk = ee.Feature(failed_chunk_polygon)
                estimated_tiles = self.net(failed_chunk.geometry(), collection, band, tile_size_m, name_title_fn, vectoriser=self.vectoriser, full_only=flag_full_chunks_only)

                # tag estimated tiles
                estimated_tiles = estimated_tiles.map(lambda f: f.set('tile_status', 'estimated'))

                # append estimated tiles to fishnet tiles
                fishnet_tiles.append(estimated_tiles)

        return fishnet_tiles, failed_polygons, chunks
    
    def create_tiles(self,
                     roi, 
                     tile_size_m=50,
                     chunk_size_multi=100,
                     collection="COPERNICUS/S2", 
                     band="B4",
                     full_tiles_only=False,
                     full_chunks_only=False,
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
            name_title_fn=name_tile_fn,
            flag_allow_estimated_tiles=True,  # keep true 
            flag_full_tiles_only=full_tiles_only,
            flag_full_chunks_only=full_chunks_only
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