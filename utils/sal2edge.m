data_root = '/media/nercms/NERCMS/Dataset/SOD/TestDataset/GT/MSRA-B-test/';
out_root = '/media/nercms/NERCMS/Dataset/SOD/TestDataset/Edge/MSRA-B-test/';

if ~exist(out_root, 'dir')
        mkdir(out_root);
end
 
imgFiles = dir([data_root '*.png']);  
imgNUM = length(imgFiles);

for im_id = 1:imgNUM

    id = imgFiles(im_id).name

    gt = imread(fullfile(data_root, id));
    %gt = (gt > 128);
    gt = double(gt);

    [gy, gx] = gradient(gt);
    temp_edge = gy.*gy + gx.*gx;
    temp_edge(temp_edge~=0)=1;
    bound = uint8(temp_edge*255);

    save_path = fullfile(out_root, id);
    imwrite(bound, save_path);

end