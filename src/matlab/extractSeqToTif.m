for i = [1,2,3,4,5,6,7,8,9]
seqFile = append('C:/Users/Henkka/Projects/fpc-diffrend/data/calibration/01/', 'pod3colour_000', string(i), '.seq');
outFile = append('C:/Users/Henkka/Projects/fpc-diffrend/data/calibration/extracted/', 'pod3colour_000', string(i), '.tif');
myImg = ReadJpegSEQ(convertStringsToChars(seqFile));
imwrite(myImg{1}, convertStringsToChars(outFile))
end