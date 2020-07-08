function [ result ] = weight(I, w, ty, tx, sigma)

    % Calculate Is' the of propagated image filter
    % input arguments:
    % I: Input image
    % w: Manhattan distance
    % ty: y coordinate of pixel s
    % tx: x coordinate of pixel s
    % sigma

    [m, n] = size(I);

    zs = 0; % preallocate a value to zs
    % Calculate the weight of the center point s
    centerweight = 1; % wss = 1
    weightmatrix = zeros(2 * w + 1, 2 * w + 1);
    weightmatrix(w + 1, w + 1) = centerweight; 
    totalweightedsum = centerweight * I(ty, tx); % wss*It of center point

    % Calculate distance from 1 to w
    for r = 1:w

        for dp = 0:r

            for psign = [-1 1] % psign = -1, +1

                p = psign * dp;

                for qsign = [-1 1] % qsign = -1, +1

                    q = qsign * (r - dp);

                    % check boundary
                    if ty + p < 1 || ty + p > m  || tx + q < 1 || tx + q > n 

                        continue

                    end

                    % decide parent pixel t-1 (t_1_y & t_1_x are coordinate shifts w.r.t. ty,tx)

                    if (p * q == 0) % on the x or y axis

                        if (p == 0)
                            t_1_y = p;
                            t_1_x = q - qsign;

                        else % q == 0
                            t_1_y = p - psign;
                            t_1_x = q;

                        end

                    else % p*q != 0 (other pixels)

                        if ( mod(r, 2) ~= 0 ) % if r is odd -> p, else -> q
                            t_1_y = p;
                            t_1_x = q - qsign;

                        else
                            t_1_y = p - psign;
                            t_1_x = q;

                        end

                    end

                    % calculate log weight
                    logweight_D = -1 * (I(ty + p, tx + q) - I(t_1_y + ty, t_1_x + tx))^2 / (2 * sigma^2);
                    logweight_R = -1 * (I(ty + p, tx + q) - I(ty, tx))^2 / (2 * sigma^2);
                    weight = exp(logweight_D + logweight_R); % weight = D(t-1,t)*R(s,t)

                    weightmatrix(w + 1 + p,  w+ 1 + q) = weight; % store D*R in this matrix 

                    wst = weight * weightmatrix(w + 1 + t_1_y, w + 1 + t_1_x); % wst = wst-1*D(t-1,t)*R(s,t) (eq.7 in the paper)
                    zs = zs + weight; % sigma wst (excluding center weight wss)
                    totalweightedsum = totalweightedsum + wst * I(ty + p, tx + q); % sigma wst*It

                    % ensure pixels on the x or y axis are calculated only one time
                    if (q == 0)
                        break
                    end  

                end

                % ensure pixels on the axis is calculated only one time
                if (p == 0)
                    break
                end

            end

        end

    end

    % ensure centerweight wss is added only once
    zs = zs + centerweight;

    % Calculate result pixel value Is' (eq.6 in the paper)
    result = totalweightedsum / zs;
    clear totalweightedsum

end
