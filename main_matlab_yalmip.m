PATH_YALMIP = "/home/marco/Documents/MATLAB/Add-Ons/Toolboxes/YALMIP";

addpath(genpath(PATH_YALMIP))

% Discrete time model of a quadcopter
Ad = [1       0       0   0   0   0   0.1     0       0    0       0       0;
      0       1       0   0   0   0   0       0.1     0    0       0       0;
      0       0       1   0   0   0   0       0       0.1  0       0       0;
      0.0488  0       0   1   0   0   0.0016  0       0    0.0992  0       0;
      0      -0.0488  0   0   1   0   0      -0.0016  0    0       0.0992  0;
      0       0       0   0   0   1   0       0       0    0       0       0.0992;
      0       0       0   0   0   0   1       0       0    0       0       0;
      0       0       0   0   0   0   0       1       0    0       0       0;
      0       0       0   0   0   0   0       0       1    0       0       0;
      0.9734  0       0   0   0   0   0.0488  0       0    0.9846  0       0;
      0      -0.9734  0   0   0   0   0      -0.0488  0    0       0.9846  0;
      0       0       0   0   0   0   0       0       0    0       0       0.9846];
Bd = [0      -0.0726  0       0.0726;
     -0.0726  0       0.0726  0;
     -0.0152  0.0152 -0.0152  0.0152;
      0      -0.0006 -0.0000  0.0006;
      0.0006  0      -0.0006  0;
      0.0106  0.0106  0.0106  0.0106;
      0      -1.4512  0       1.4512;
     -1.4512  0       1.4512  0;
     -0.3049  0.3049 -0.3049  0.3049;
      0      -0.0236  0       0.0236;
      0.0236  0      -0.0236  0;
      0.2107  0.2107  0.2107  0.2107];
[nx, nu] = size(Bd);

% Constraints
u0 = 10.5916;
umin = [9.6; 9.6; 9.6; 9.6] - u0;
umax = [13; 13; 13; 13] - u0;
xmin = [-pi/6; -pi/6; -Inf; -Inf; -Inf; -1; -Inf(6,1)];
xmax = [ pi/6;  pi/6;  Inf;  Inf;  Inf; Inf; Inf(6,1)];

% Objective function
Q = diag([0 0 10 10 10 10 0 0 0 5 5 5]);
QN = Q;
R = 0.1*eye(4);

% Initial and reference states
x0 = zeros(12,1);
xr = [0; 0; 1; 0; 0; 0; 0; 0; 0; 0; 0; 0];

% Prediction horizon
N = 10;

% Define problem
u = sdpvar(repmat(nu,1,N), repmat(1,1,N));
x = sdpvar(repmat(nx,1,N+1), repmat(1,1,N+1));
constraints = [xmin <= x{1} <= xmax];
objective = 0;
for k = 1 : N
    objective = objective + (x{k}-xr)'*Q*(x{k}-xr) + u{k}'*R*u{k};
    constraints = [constraints, x{k+1} == Ad*x{k} + Bd*u{k}];
    constraints = [constraints, umin <= u{k}<= umax, xmin <= x{k+1} <= xmax];
end
objective = objective + (x{N+1}-xr)'*QN*(x{N+1}-xr);
options = sdpsettings('solver', 'gurobi');
controller = optimizer(constraints, objective, options, x{1}, [u{:}]);

% Simulate in closed loop
nsim = 15;
for i = 1 : nsim
    U = controller{x0};
    x0 = Ad*x0 + Bd*U(:,1);
end
