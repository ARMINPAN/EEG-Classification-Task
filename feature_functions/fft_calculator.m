function [f, P1] = fft_calculator(data, Fs)
    % calculate single side band fft of a vector signal
    L = size(data,1);
    fft_data = fft(data);
    P2 = abs(fft_data/L);
    P1 = P2(1:L/2+1);
    P1(2:end-1) = 2*P1(2:end-1);
    f = Fs*(0:(L/2))/L;
end