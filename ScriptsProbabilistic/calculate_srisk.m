% [INPUT]
% lrmes = A vector of floats containing the Long Run MES values.
% tl_x  = A numeric vector containing the firm total liabilities.
% mc_x  = A numeric vector containing the firm market capitalization.
% l     = A float representing the capital adequacy ratio (optional, default=0.08).
%
% [OUTPUT]
% srisk = A vector of floats containing the SRISK values.

function srisk = calculate_srisk(varargin)

    persistent ip;

    if (isempty(ip))
        ip = inputParser();
        ip.addRequired('lrmes',@(x)validateattributes(x,{'double','single'},{'vector','finite','nonempty','nonnan','real'}));
        ip.addRequired('tl_x',@(x)validateattributes(x,{'numeric'},{'vector','finite','nonempty','nonnan','real'}));
        ip.addRequired('mc_x',@(x)validateattributes(x,{'numeric'},{'vector','finite','nonempty','nonnan','real'}));
        ip.addOptional('l',0.08,@(x)validateattributes(x,{'double','single'},{'scalar','real','finite','>=',0.05,'<=',0.20}));
    end

    ip.parse(varargin{:});
    ip_res = ip.Results;

    srisk = calculate_srisk_internal(ip_res.lrmes,ip_res.tl_x,ip_res.mc_x,ip_res.l);

end

function srisk = calculate_srisk_internal(lrmes,tl_x,mc_x,l)

    srisk = (l .* tl_x) - ((1 - l) .* (1 - lrmes) .* mc_x);
    srisk(srisk<0) = 0;

end
