import unittest

import ee

import sankee

ee.Initialize()


class TestDatasets(unittest.TestCase):
    def test_get_year_nlcd(self):
        dataset = sankee.datasets.NLCD
        img = dataset.get_year(2016)
        self.assertEqual(img.get("system:id").getInfo(), "USGS/NLCD_RELEASES/2019_REL/NLCD/2016")
        self.assertListEqual(img.bandNames().getInfo(), [dataset.band])

    def test_get_year_LCMS_LC(self):
        dataset = sankee.datasets.LCMS_LC
        img = dataset.get_year(2016)
        self.assertEqual(
            img.get("system:id").getInfo(), "USFS/GTAC/LCMS/v2021-7/LCMS_CONUS_v2021-7_2016"
        )
        self.assertListEqual(img.bandNames().getInfo(), [dataset.band])

    def test_get_year_LCMS_LU(self):
        dataset = sankee.datasets.LCMS_LU
        img = dataset.get_year(2016)
        self.assertEqual(
            img.get("system:id").getInfo(), "USFS/GTAC/LCMS/v2021-7/LCMS_CONUS_v2021-7_2016"
        )
        self.assertListEqual(img.bandNames().getInfo(), [dataset.band])

    def test_get_year_CGLS(self):
        dataset = sankee.datasets.CGLS_LC100
        img = dataset.get_year(2016)
        self.assertEqual(
            img.get("system:id").getInfo(), "COPERNICUS/Landcover/100m/Proba-V-C3/Global/2016"
        )
        self.assertListEqual(img.bandNames().getInfo(), [dataset.band])

    def test_get_year_MODIS(self):
        datasets = [
            sankee.datasets.MODIS_LC_TYPE1,
            sankee.datasets.MODIS_LC_TYPE2,
            sankee.datasets.MODIS_LC_TYPE3,
        ]

        for dataset in datasets:
            img = dataset.get_year(2016)
            self.assertEqual(img.get("system:id").getInfo(), "MODIS/006/MCD12Q1/2016_01_01")
            self.assertListEqual(img.bandNames().getInfo(), [dataset.band])

    def test_years(self):
        for dataset in sankee.datasets.datasets:
            self.assertListEqual(dataset.years, dataset.list_years().getInfo())

    def test_get_unsupported_year(self):
        with self.assertRaisesRegex(ValueError, "does not include year"):
            sankee.datasets.NLCD.get_year(2017)

    def test_get_invalid_years(self):
        """Single or duplicate years should raise errors."""
        with self.assertRaisesRegex(ValueError, "at least two years"):
            sankee.datasets.LCMS_LU.sankify(years=[2017], region=None)
        with self.assertRaisesRegex(ValueError, "Duplicate years"):
            sankee.datasets.LCMS_LU.sankify(years=[2017, 2017, 2018], region=None)
