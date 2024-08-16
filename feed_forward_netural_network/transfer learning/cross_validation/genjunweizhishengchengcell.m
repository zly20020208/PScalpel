% Load the mutations data from the MAT file
load('cGAS_mutations.mat', 'mutations');

% Assume original_sequence is your original amino acid sequence string
original_sequence = 'QPWHGKAMQRASEAGATAPKASARNARGAPMDPTESPAAPEAALPKAGKFGPARKSGSRQKKSAPDTQERPPVRATGARAKKAPQRAQDTQPSDATSAPGAEGLEPPAAREPALSRAGSCRQRGARCSTKPRPPPGPWDVPSPGLPVSAPILVRRDAAPGASKLRAVLEKLKLSRDDISTAAGMVKGVVDHLLLRLKCDSAFRGVGLLNTGSYYEHVKISAPNEFDVMFKLEVPRIQLEEYSNTRAYYFVKFKRNPKENPLSQFLEGEILSASKMLSKFRKIIKEEINDIKDTDVIMKRKRGGSPAVTLLISEKISVDITLALESKSSWPASTQEGLRIQNWLSAKVRKQLRLKPFYLVPKHAKEGNGFQEETWRLSFSHIEKEILNNHGKSKTCCENKEEKCCRKDCLKLMKYLLEQLKERFKDKKHLDKFSSYHVKTAFFHVCTQNPQDSQWDRKDLGLCFDNCVTYFLQCLRTEKLENYFIPEFNLFSSNLIDKRSKEFLTKQIEYERNNEFPVFDEF'; % 应该包含全部氨基酸序列
% Initialize containers for mutated sequences based on categories
mutations_yes = {};
mutations_no = {};

% Process each mutation
for i = 1:size(mutations, 1)
    mutation_info = mutations{i, 1};  % e.g., 'D169G'
    category = mutations{i, 2};  % e.g., '++'

    % Parse the mutation information
    original_aa = mutation_info(1);  % Original amino acid
    position = str2double(mutation_info(2:end-1));  % Position of the mutation
    new_aa = mutation_info(end);  % New amino acid

    % Create the mutated sequence
    mutated_sequence = original_sequence;
    if new_aa == '-'
        mutated_sequence(position) = [];
    else
        mutated_sequence(position) = new_aa;
    end

    % Categorize the mutated sequence
    switch category
        case '+'
            mutations_yes{end+1} = mutated_sequence;
        case '/'
            mutations_no{end+1} = mutated_sequence;
    end
end

% Save the categorized mutated sequences into separate MAT files
save('CGAS_mutations_yes.mat', 'mutations_yes');
save('CGAS_mutations_no.mat', 'mutations_no');
