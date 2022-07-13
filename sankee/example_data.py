import ee


# These can only be loaded once ee is initialized, which is why we put these in a class
class regions:
    hobet_coal_mine = ee.Geometry.Polygon(
        [
            [
                [-81.973301, 38.121953],
                [-82.000086, 38.105746],
                [-82.009701, 38.077104],
                [-82.000429, 38.060617],
                [-81.965403, 38.043044],
                [-81.950637, 38.073321],
                [-81.942052, 38.107907],
                [-81.973301, 38.121953],
            ]
        ]
    )

    mt_st_helens = ee.Geometry.Polygon(
        [
            [
                [-122.289907, 46.289229],
                [-122.22328, 46.329545],
                [-122.15528, 46.346138],
                [-122.053623, 46.323855],
                [-122.033704, 46.252682],
                [-122.102391, 46.231786],
                [-122.161462, 46.24176],
                [-122.260371, 46.233686],
                [-122.289907, 46.289229],
            ]
        ]
    )

    las_vegas = ee.Geometry.Polygon(
        [
            [
                [-115.01184401606046, 36.24170785506492],
                [-114.98849806879484, 36.29928186470082],
                [-115.25628981684171, 36.35238941394592],
                [-115.34692702387296, 36.310348922031565],
                [-115.37988600824796, 36.160811202271944],
                [-115.30298171137296, 36.03653336474891],
                [-115.25628981684171, 36.05207884201088],
                [-115.26590285395109, 36.226199908103695],
                [-115.19174513910734, 36.25499793268206],
            ]
        ]
    )

    rim_fire = (
        ee.FeatureCollection("projects/sat-io/open-datasets/MTBS/burned_area_boundaries")
        .filter(ee.Filter.eq("Event_ID", "CA3785712008620130817"))
        .first()
        .geometry()
    )