% Define the degree of the polynomial
n = 7;

% Define ranges for a and b
a_values = [1, 2, 3, 4, 6, 9];
b_values = [1, 2, 3, 4, 6, 9];

% Define the x values for plotting
x = linspace(-1, 1, 100);

% Create a new figure window
figure;

% Plot each combination of a and b
plotNum = 1; % To manage subplot indexing
for a = a_values
    for b = b_values
        % Compute the Jacobi polynomial using symbolic variables
        P = jacobiP(n, a, b, x);
        
        % Create a subplot for each combination
        subplot(length(a_values), length(b_values), plotNum);
        plot(x, P);
        grid on;
        title(sprintf('Jacobi P(%d, %d, %d, x)', n, a, b));
        xlabel('x');
        ylabel(sprintf('P_{%d}^{(%d,%d)}(x)', n, a, b));
        
        % Increment the subplot index
        plotNum = plotNum + 1;
    end
end

% Adjust layout to prevent subplot titles and labels from overlapping
sgtitle('Jacobi Polynomials for Various a and b');
set(gcf, 'Position', [100, 100, 700, 500]); % Set figure size
