#!/bin/sh
# echo "23년 6월 공산성"
# python create_h5_data.py --shp_file_path ../shp/02_2023년_6월_공산성 --shp_file_name GS_23_6_Akk_Polygon.shp --hsp_img_path ../img/gongsan --hsp_img_name 03_20230603_Gongsan_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202306_gongsan.h5
# python create_h5_data.py --akk_shp_file_path ../shp/02_2023년_6월_공산성 --akk_shp_file_name GS_23_6_Akk_subset_Polygon.shp --hsp_img_path ../img/gongsan --hsp_img_name 03_20230603_Gongsan_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202306_gongsan.h5
# python create_h5_data.py --akk_shp_file_path ../shp/02_2023년_6월_공산성 --akk_shp_file_name GS_23_6_Akk_subset_Polygon.shp --hsp_img_path ../img/gongsan --hsp_img_name 03_20230603_Gongsan_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202306_downsampled_gongsan.h5

# echo "23년 7월 공산성"
# python create_h5_data.py --shp_file_path ../shp/03_2023년_7월_공산성 --shp_file_name GS_23_7_Akk_Polygon.shp --hsp_img_path ../img/gongsan --hsp_img_name 03_20230703_Gongsan_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202307_gongsan.h5
# python create_h5_data.py --akk_shp_file_path ../shp/03_2023년_7월_공산성 --akk_shp_file_name GS_23_7_Akk_subset_Polygon.shp --gt_shp_file_name GS_23_7_gt_Polygon.shp --hsp_img_path ../img/gongsan --hsp_img_name 03_20230703_Gongsan_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202307_gongsan.h5
python create_h5_data2.py --akk_shp_file_path ../shp/03_2023년_7월_공산성 --akk_shp_file_name GS_23_7_Akk_subset_Polygon.shp --gt_shp_file_name GS_23_7_gt_Polygon.shp --hsp_img_path ../img/gongsan --hsp_img_name 03_20230703_Gongsan_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202307_downsampled_gongsan.h5 --downsample_factor 2
python create_h5_data2.py --akk_shp_file_path ../shp/03_2023년_7월_공산성 --akk_shp_file_name GS_23_7_Akk_subset_Polygon.shp --gt_shp_file_name GS_23_7_gt_Polygon.shp --hsp_img_path ../img/gongsan --hsp_img_name 03_20230703_Gongsan_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202307_downsampled_gongsan.h5 --downsample_factor 4
python create_h5_data2.py --akk_shp_file_path ../shp/03_2023년_7월_공산성 --akk_shp_file_name GS_23_7_Akk_subset_Polygon.shp --gt_shp_file_name GS_23_7_gt_Polygon.shp --hsp_img_path ../img/gongsan --hsp_img_name 03_20230703_Gongsan_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202307_downsampled_gongsan.h5 --downsample_factor 8

# echo "23년 8월 공산성"
# python create_h5_data.py --shp_file_path ../shp/04_2023년_8월_공산성 --shp_file_name GS_23_8_Akk_Polygon.shp --hsp_img_path ../img/gongsan --hsp_img_name 03_20230803_Gongsan_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202308_gongsan.h5
# python create_h5_data.py --akk_shp_file_path ../shp/04_2023년_8월_공산성 --akk_shp_file_name GS_23_8_Akk_subset_Polygon.shp --gt_shp_file_name GS_23_8_gt_Polygon.shp --hsp_img_path ../img/gongsan --hsp_img_name 03_20230803_Gongsan_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202308_gongsan.h5
# python create_h5_data.py --akk_shp_file_path ../shp/04_2023년_8월_공산성 --akk_shp_file_name GS_23_8_Akk_subset_Polygon.shp --gt_shp_file_name GS_23_8_gt_Polygon.shp --hsp_img_path ../img/gongsan --hsp_img_name 03_20230803_Gongsan_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202308_downsampled_gongsan.h5

# echo "23년 6월 부소산성"
# # # python create_h5_data.py --shp_file_path ../shp/06_2023년_6월_부소산성 --shp_file_name BS_23_6_AKK_Polygon.shp --hsp_img_path ../img/buso --hsp_img_name 202306_Buso_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202306_buso.h5
# # python create_h5_data.py --akk_shp_file_path ../shp/06_2023년_6월_부소산성 --akk_shp_file_name BS_23_6_AKK_Polygon.shp --gt_shp_file_name BS_23_6_gt_Polygon.shp --hsp_img_path ../img/buso --hsp_img_name 202306_Buso_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202306_buso.h5
# python create_h5_data.py --akk_shp_file_path ../shp/06_2023년_6월_부소산성 --akk_shp_file_name BS_23_6_AKK_Polygon.shp --gt_shp_file_name BS_23_6_gt_Polygon.shp --hsp_img_path ../img/buso --hsp_img_name 202306_Buso_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202306_downsampled_buso.h5

# echo "23년 7월 부소산성"
# # # python create_h5_data.py --shp_file_path ../shp/07_2023년_7월_부소산성 --shp_file_name BS_23년_7월_AKK_Polygon2.shp --hsp_img_path ../img/buso --hsp_img_name 202307_Buso_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202307_buso.h5
# # python create_h5_data.py --akk_shp_file_path ../shp/07_2023년_7월_부소산성 --akk_shp_file_name BS_23년_7월_AKK_Polygon2.shp --gt_shp_file_name BS_23_7_gt_Polygon.shp --hsp_img_path ../img/buso --hsp_img_name 202307_Buso_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202307_buso.h5
# python create_h5_data.py --akk_shp_file_path ../shp/07_2023년_7월_부소산성 --akk_shp_file_name BS_23년_7월_AKK_Polygon2.shp --gt_shp_file_name BS_23_7_gt_Polygon.shp --hsp_img_path ../img/buso --hsp_img_name 202307_Buso_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202307_downsampled_buso.h5

# echo "23년 8월 부소산성"
# # # python create_h5_data.py --shp_file_path ../shp/08_2023년_8월_부소산성 --shp_file_name BS_23_8_AKK_Polygon.shp --hsp_img_path ../img/buso --hsp_img_name 202308_Buso_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202308_buso.h5
# # python create_h5_data.py --akk_shp_file_path ../shp/08_2023년_8월_부소산성 --akk_shp_file_name BS_23_8_AKK_Polygon.shp --gt_shp_file_name BS_23_8_gt_Polygon.shp --hsp_img_path ../img/buso --hsp_img_name 202308_Buso_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202308_buso.h5
# python create_h5_data.py --akk_shp_file_path ../shp/08_2023년_8월_부소산성 --akk_shp_file_name BS_23_8_AKK_Polygon.shp --gt_shp_file_name BS_23_8_gt_Polygon.shp --hsp_img_path ../img/buso --hsp_img_name 202308_Buso_Hyper_Subset --output_h5_path ../h5_data --output_h5_file 202308_downsampled_buso.h5
# python demo_bigdata.py --dataset='GS' --epoches=100 --patches=1 --band_patches=3 --cache_path=./prs_data/cache_data2 --mode='CAF' --seed=123123

python create_h5_data2.py --akk_shp_file_path ../shp/GS_24_7_Akk_Polygon --akk_shp_file_name GS_24_7_Akk_Polygon.shp --gt_shp_file_name GS_24_7_gt_Polygon.shp --hsp_img_path ../img/gongsan --hsp_img_name 00_20240730_GS_Mosaic_Subset --output_h5_path ../h5_data --output_h5_file 202407_downsampled_gongsan.h5 --downsample_factor 2
python create_h5_data2.py --akk_shp_file_path ../shp/GS_24_7_Akk_Polygon --akk_shp_file_name GS_24_7_Akk_Polygon.shp --gt_shp_file_name GS_24_7_gt_Polygon.shp --hsp_img_path ../img/gongsan --hsp_img_name 00_20240730_GS_Mosaic_Subset --output_h5_path ../h5_data --output_h5_file 202407_downsampled_gongsan.h5 --downsample_factor 4
python create_h5_data2.py --akk_shp_file_path ../shp/GS_24_7_Akk_Polygon --akk_shp_file_name GS_24_7_Akk_Polygon.shp --gt_shp_file_name GS_24_7_gt_Polygon.shp --hsp_img_path ../img/gongsan --hsp_img_name 00_20240730_GS_Mosaic_Subset --output_h5_path ../h5_data --output_h5_file 202407_downsampled_gongsan.h5 --downsample_factor 8