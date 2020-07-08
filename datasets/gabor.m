function [ net ] = gabor(dataSetName) 

% clc;
% clear all;


switch dataSetName
  case 'Indianpines' 
        load('../datasets/Indian_pines_corrected.mat');load('../datasets/Indian_pines_gt.mat');
        x = indian_pines_corrected;y = indian_pines_gt;
    case 'KSC'      
        load('../datasets/KSC.mat');load('../datasets/KSC_gt.mat');
        x = KSC;y = KSC_gt;
    case 'Salinas'    
        load('../datasets/Salinas_corrected.mat');load('../datasets/Salinas_gt.mat');
        x = salinas_corrected;y = salinas_gt;
    case 'SalinasA' 
     load('../datasets/SalinasA_corrected.mat');load('../datasets/SalinasA_gt.mat');
     x = salinasA_corrected; y = salinasA_gt;
    case 'Pavia' 
     load('../datasets/Pavia.mat');load('../datasets/Pavia_gt.mat');
     x = pavia; y = pavia_gt;
    case 'PaviaU' 
     load('../datasets/PaviaU.mat');load('../datasets/PaviaU_gt.mat');
     x = paviaU; y = paviaU_gt;
  case 'Botswana'
        load('../datasets/Botswana.mat');load('../datasets/Botswana_gt.mat');
        x = Botswana;y = Botswana_gt;
    case 'Urban4'
        load('../datasets/Urban_R162.mat');load('../datasets/end4_groundTruth.mat');
        Urban = reshape(Y', 307,307,162); [uMax, Urban_gt] =max(A); Urban_gt = reshape(Urban_gt, 307,307);
        Urban_gt(307, 307) = 0;             %to be compatible  with  nClass = size(ind,1)-1; 
        x = Urban;y = Urban_gt;
    case 'Urban5'
        load('../datasets/Urban_R162.mat');load('../datasets/end5_groundTruth.mat');
        Urban = reshape(Y', 307,307,162); [uMax, Urban_gt] =max(A); Urban_gt = reshape(Urban_gt, 307,307);
        Urban_gt(307, 307) = 0;             %to be compatible  with  nClass = size(ind,1)-1; 
        x = Urban;y = Urban_gt;
    case 'Urban6'
        load('../datasets/Urban_R162.mat');load('../datasets/end6_groundTruth.mat');
        Urban = reshape(Y', 307,307,162); [uMax, Urban_gt] =max(A); Urban_gt = reshape(Urban_gt, 307,307);
        Urban_gt(307, 307) = 0;             %to be compatible  with  nClass = size(ind,1)-1; 
        x = Urban;y = Urban_gt;
    case 'Indianpines5' 
        load('../datasets/ndian_pines_5class.mat');
%         x = indian_pines_corrected;y = indian_pines_gt;
    case 'Earthquake'
        addpath('E:/History Matching/EarthquakeData');
        load('well_corrected.mat','well_corrected');load('well_gt.mat','well_gt');
        x = well_corrected;y = well_gt;
        clear well_corrected well_gt;        
    otherwise
        error('Unknown data set requested.');
end



DataSet = single(x);


[ nRow, nColumn, nBand ] = size( DataSet );
HalfWidth=13;
DataSet = 1 * ( ( DataSet - min( DataSet( : ) ) ) / ( max( DataSet( : ) ) - min( DataSet( : ) ) )-0.5);

Data=DataSet;
nPC = 3;  
nSample = nRow * nColumn;
Data = reshape( Data, nRow * nColumn, nBand );
DataMean = mean( Data );

C = zeros( nBand, nBand );
C = Data' * Data / nSample - ( DataMean' ) * DataMean;
C = double( C );
[ V, D ] = eigs( C, nPC );

epsilon=1e-5;

PC =  ( Data - repmat( DataMean, [ nSample, 1 ] ) ) *V*diag(1./sqrt(diag(D)+epsilon));
PCData = reshape( PC, nRow, nColumn, nPC );


za=pickfre(0.2,0);
zb=pickfre(0.2,pi/4);
zc=pickfre(0.2,pi/2);
zd=pickfre(0.2,3*pi/4);

DATA=[];
F=[];
    for i=1:nPC
      fa =filter2(za,PCData(:,:,i),'valid');  
      A = abs(fa);  
      fb =filter2(zb,PCData(:,:,i),'valid');  
      B = abs(fb);
      fc =filter2(zc,PCData(:,:,i),'valid'); 
      C = abs(fc);
      fd =filter2(zd,PCData(:,:,i),'valid'); 
      D=abs(fd);
      F=cat(3,F,A,B,C,D);
    end
DATA=cat(4,DATA,F);
save_name=strcat('gabor_data_',dataSetName,'.mat');
save(save_name, 'DATA')
% save('gabor_data.mat', 'DATA')


function z=pickfre(f,theta)
z=zeros(3,3);
x = 0;  
    for i = linspace(-2,2,3)  
        x = x + 1;  
        y = 0;  
        for j = linspace(-2,2,3)  
            y = y + 1;  
            z(y,x)=compute(i,j,f,theta);  
        end  
    end  

    
    
function gabor_k = compute(x,y,f0,theta)  
r = 1; g = 1;  
x1 = x*cos(theta) + y*sin(theta);  
y1 = -x*sin(theta) + y*cos(theta);  
gabor_k = f0^2/(pi*r*g)*exp(-(f0^2*x1^2/r^2+f0^2*y1^2/g^2))*exp(1i*2*pi*f0*x1);  
