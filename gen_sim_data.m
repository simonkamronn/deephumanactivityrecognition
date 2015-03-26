close all; clear all


NCLASS=6;
NFEAT=6;
SEQLEN=128;
NSAMPLES = 2500;
NOISE=1.5;

CLASS = round(rand(NSAMPLES,1)*NCLASS+0.5);
TARGET = zeros(NSAMPLES,NCLASS);
for i = 1:NSAMPLES
    TARGET(i,CLASS(i)) = 1;
end

INPUT = randn(NSAMPLES,NFEAT,SEQLEN)*NOISE;
SIGNAL = sin(linspace(0,20,SEQLEN*2)');

for i = 1:NSAMPLES
    OFFSET = randi(SEQLEN,1); %samples an int between 1:SEQLEN 
    INPUT(i,CLASS(i),:) = squeeze(INPUT(i,CLASS(i),:)) + SIGNAL(OFFSET:(SEQLEN+OFFSET-1));
end


%random sample an off set

split1 = 1:500;
split2 = 501:1000;
split3 = 1001:1500;
split4 = 1501:2000;
split5 = 2001:2500;

input1 = INPUT(split1,:,:);
input2 = INPUT(split2,:,:);
input3 = INPUT(split3,:,:);
input4 = INPUT(split4,:,:);
input5 = INPUT(split5,:,:);

target1 = TARGET(split1,:,:);
target2 = TARGET(split2,:,:);
target3 = TARGET(split3,:,:);
target4 = TARGET(split4,:,:);
target5 = TARGET(split5,:,:);

save('data/simdata.mat','input*','target*','-v6')
