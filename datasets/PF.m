function [] =  SVMPCAPF(dataSetName, sigma, w, dimen)

    sigma=str2num(sigma)
    w=str2num(w)
    dimen=str2num(dimen)
    %%%% load the ground truth and the hyperspectral image
    switch dataSetName
        case 'Indianpines' 
            load('Indian_pines_corrected.mat');load('Indian_pines_gt.mat');
            x = indian_pines_corrected;y = indian_pines_gt;
        case 'KSC'      
            load('KSC.mat');load('KSC_gt.mat');
            x = KSC;y = KSC_gt;
        case 'Salinas'    
            load('Salinas_corrected.mat');load('Salinas_gt.mat');
            x = salinas_corrected;y = salinas_gt;
        case 'SalinasA' 
           load('SalinasA_corrected.mat');load('SalinasA_gt.mat');
           x = salinasA_corrected; y = salinasA_gt;
        case 'Pavia' 
           load('Pavia.mat');load('Pavia_gt.mat');
           x = pavia; y = pavia_gt;
        case 'PaviaU' 
           load('PaviaU.mat');load('PaviaU_gt.mat');
           x = paviaU; y = paviaU_gt;
        otherwise
            error('Unknown data set requested.');
    end        
      
    gt=double(y);
    img=x;
    img=img./max(max(max(img)));

    %%% size of image 
    [no_lines, no_rows, no_bands] = size(img);

    %%% image fusion
    img2=average_fusion(img,dimen);
    %% normalization
    no_bands=size(img2,3);
    fimg=reshape(img2,[no_lines*no_rows no_bands]);
    [fimg] = scale_new(fimg);
    fimg=reshape(fimg,[no_lines no_rows no_bands]);
    %% feature construction
    fimg=spatial_feature(fimg,w,sigma);

    save_name=strcat(dataSetName,'_PF.mat');
    save(save_name, 'fimg');
end
