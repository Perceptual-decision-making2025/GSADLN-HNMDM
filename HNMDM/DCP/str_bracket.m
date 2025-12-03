function str2 = str_bracket( str,flag )

if nargin==1
    left = '[';
    right = ']';  
elseif nargin==2 
    if ischar(flag)
        if length(flag) == 2
           left = flag(1);
           right = flag(2);
        else
            error('The length of the flag string must be 2');
        end
    else
        if any( flag == [0 1 2] )
            switch flag
                case 0
                    left = '(';
                    right = ')';
                case 1
                    left = '[';
                    right = ']';
                case 2
                    left = '{';
                    right = '}';
            end
        else
            error('The value of the flag must be a string of length 2 or the numerical values 0, 1, or 2');
        end
    end
end

[m,n] = size(str);
str2 = [];
for ii = 1:m
    if ii == 1
        str_temp = strcat(left,str(1,:));
    elseif ii == m
        str_temp = strcat(str(m,:),right);
    else
        str_temp = str(ii,:);
    end
    str2 = strvcat(str2,str_temp);
end

if m ==1  
     str2 = strcat(str2,right);
end

                
        
    
