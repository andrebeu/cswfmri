

findZeros = find(data==0);
data(findZeros) = 10;

subplot(2,1,1),imagesc(data(2:13,:))
colormap colorcube
box off
caxis([1,10])

subplot(2,1,2),imagesc(data(16:27,:))
colormap colorcube
box off
caxis([1,10])