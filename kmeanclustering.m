clc; clear; close all;

%% Read image

%Original Image
im=imread("mandrill.jpg");
imagefloat=double(im);
figure(1);imshow(im);

typeOfK=0; %0= k means clustering %1=k++mean clustering
k=4;

%Choose 4 pixels
szIm=size(im);
numPx=szIm(1)*szIm(2);

%initialisation
dist = zeros([szIm(1:2) k]);% distance from the mean
assign = ones(szIm(1:2));   % assignement of the pixels
imOut = zeros(szIm);        % the colour coded output image


scatterRows=1:szIm(1);
scatterCols=1:100:szIm(2);
scatterPoints=zeros(length(scatterRows)*length(scatterCols),3);


%% Finding the centre
rng('shuffle');
if (typeOfK)
    ctrds=zeros(k,2);
    ctrds(1,:)=[randperm(szIm(1),1),randperm(szIm(2),1)];
    ctrdsLin=zeros(k,1);
    ctrdsLin(1)=sub2ind(szIm(1:2),ctrds(1,1),ctrds(1,2));
    
    mu=zeros(k,1,3); %mean values
    
    d = 1e3*ones(szIm(1),szIm(2),k-1);  % distance vector
    
    for i=1:k-1
        mu(i,1,:)=imagefloat(ctrds(i,1),ctrds(i,2),:);
        d(:,:,i) = sqrt(sum((imagefloat - mu(i,1,:)).^2,3)); % distance
        Dx=min(d,[],3);
        Dx=reshape(Dx,[],1);
        idxRange=ceil(0.75*numPx:numPx);
        [bigDx,oriDx]= sort(Dx);
        bigDx=bigDx(idxRange);
        oriDx=oriDx(idxRange);
        prob=bigDx/sum(bigDx); %probabilty of each points
        Cprob=cumsum(prob); %cumulative probability
        ctrdsLin(i+1)=oriDx(find(Cprob>=rand(),1)); %update the linear index
        
        %check if the next centroid matches
        while (length(unique(ctrdsLin(1:1+i)))~= i+1)
            ctrdsLin(i+1)=ctrdsLin(i+1)+1;
        end 
        %assign new centroid
        [ctrds(i+1,1), ctrds(i+1,2)] = ind2sub(szIm(1:2),ctrdsLin(i+1));
    end 
    %update new mean
    mu(k,1,:)=imagefloat(ctrds(k,1),ctrds(k,2),:);
        
else
    
    ctrds = [randperm(szIm(1),k); randperm(szIm(2),k)]';   % random centroids for the mean
    mu = zeros(k,1,3);
    for i = 1:k
        % get the initial mean values from the image from the random points
        mu(i,1,:) = imagefloat(ctrds(i,1), ctrds(i,2), :);
    end
end


%% repeat the process 
n=0;
while(1)
    
    %variables
    n=n+1;
    assign2=assign;
    
     %Calculate RGB distance 
     for i=1:k
          dist(:,:,i) = sqrt(sum((imagefloat-mu(i,1,:)).^2,3));
     end
     
    %assign the minimum distance 
    [~, assign] = min(dist,[],3);
    
 %calculate new mean and update the output image
    for i=1:k
        clusLinIndex= find(assign==i);
    
    % gets the image values for the indices for all 3 RGB dimensions
        clusterVals = [imagefloat(clusLinIndex) imagefloat(clusLinIndex+numPx) imagefloat(clusLinIndex+2*numPx)];
        % stores the mean result in a temporary variable
        muTemp = mean(clusterVals);
        
        % stores back the result in the mu variable so that the RGB
        % components are in the 3rd dimension
        mu(i,1,:) = reshape(muTemp,[],1,3); 
        
        % Update the colour-coded output
        for j = 0:2
            
           imOut(clusLinIndex+j*numPx) = muTemp(j+1); 
           
        end
        
    end
 
 %Plotting the image
    figure(2); subplot(1,2,1); image(uint8(imOut),'CDataMapping','scaled');
    
    if (typeOfK)  
        t0=sprintf('Kmeans ++ clustering initialization:\n');
    else
        t0 = sprintf('Random initialization: \n');
    end
    t1 = sprintf('k = %d; Iteration #: %d \n',k,n); title([t0,t1,'Output image']);
    set(gca,'dataAspectRatio',[1 1 1]);
    
    %Plots scatter 3D
    for j = 1:3
           scatterPoints(:,j) = reshape(imagefloat(scatterRows,scatterCols,j),[],1);
    end
    
    scatterColors = (reshape(imOut(scatterRows,scatterCols,:),[],3))/255;
    figure(2); subplot(1,2,2); scatter3(scatterPoints(:,1),scatterPoints(:,2),scatterPoints(:,3),8,scatterColors,'filled');
    hold on; 
    scatter3(mu(:,:,1),mu(:,:,2),mu(:,:,3),40,'b','filled');
    hold off;
    title([t0,t1, 'RGB scatter plot']); xlabel('Red'); ylabel('Green'); zlabel('Blue');
    xlim([0 255]); ylim([0 255]); zlim([0 255]);
    set(gcf, 'Position', get(0, 'Screensize'));
    
    %stop when image reached 99% similarity
    if(numel(find(assign2~=assign))<0.001*numPx)
        break;
    end
    
end

%Update title
t1 = sprintf('k = %d; Done after %d iterations\n',k,n);
subplot(1,2,1); title([t0,t1, 'Output image']);  set(gca,'dataAspectRatio',[1 1 1]);
subplot(1,2,2); title([t0,t1, 'RGB scatter plot']);
    
    
  
    
