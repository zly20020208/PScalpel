function [net, adaptive_var, loss] = train_DNN1(net, parameter, adaptive_var, Y, label, global_step)
%
    dnn = net.dnn;
    L = size(dnn,2)+1;
    y = Y.Y{L};
    m = size(y,2);

    disp(['Size of y: ', mat2str(size(y))]);
    disp(['Size of label: ', mat2str(size(label))]);
    loss = sum(sum(-label.*log(y)))/m;
     g = Y.Y{L} - label;  % 初始化 g 为输出层的梯度


    for i = L : -1 : 2
        if dnn{i-1}.function == "relu"
            g = g.*(Y.Y{i} > 0);
        end
        if dnn{i-1}.function == "selu"
            g = dselu(g,Y.Y{i});
        end
        if dnn{i-1}.function == "tanh"
            g = dtanh(g,Y.Y{i});
        end
        if dnn{i-1}.function == "sigmoid"
            g = g.*Y.Y{i}.*(1 - Y.Y{i});
        end
        if dnn{i-1}.function == "softmax"
            g = Y.Y{i}-label;
        end
            [g, dgamma, dbeta] = batchnorm_backward(g, Y.Cache{i-1});
         
%  确保 Y.Cache{i-1} 包含 mu 和 var 字段
%         if isfield(Y.Cache{i-1}, 'mu') && isfield(Y.Cache{i-1}, 'var')
%         更新 dnn{i-1} 的 mu 和 var
%          dnn{i-1}.mu = Y.Cache{i-1}.mu;
%          dnn{i-1}.var = Y.Cache{i-1}.var;
%        else
%         如果 mu 和 var 不在 Y.Cache 中，可以在这里添加错误处理或警告
%         
%          disp(['mu or var not found in Y.Cache for layer ', num2str(i-1)]);
%        end
       
% 
        if isnan(dnn{i-1}.mu)
        
            dnn{i-1}.mu = Y.Cache{i-1}.mu;
            dnn{i-1}.var = Y.Cache{i-1}.var;
        else
            dnn{i-1}.mu = (1-parameter.beta1) * Y.Cache{i-1}.mu + parameter.beta1 * dnn{i-1}.mu;
            dnn{i-1}.var = (1-parameter.beta1) * Y.Cache{i-1}.var + parameter.beta1 * dnn{i-1}.var;
        end

            

            dw = g*Y.Y{i - 1}.'/m;
            db = sum(g,2)/m;
            g = dnn{i-1}.W'*g;
            adaptive_var.s{i-1}.vW = parameter.beta1*adaptive_var.s{i-1}.vW + (1 - parameter.beta1)*dw; 
            adaptive_var.s{i-1}.vb = parameter.beta1*adaptive_var.s{i-1}.vb + (1 - parameter.beta1)*db;
            adaptive_var.V{i-1}.vW = max(parameter.beta2*adaptive_var.V{i-1}.vW + (1 - parameter.beta2)*dw.*dw,adaptive_var.V{i-1}.vW); 
            adaptive_var.V{i-1}.vb = max(parameter.beta2*adaptive_var.V{i-1}.vb + (1 - parameter.beta2)*db.*db,adaptive_var.V{i-1}.vb);
            
            adaptive_var.s{i-1}.vgamma = parameter.beta1*adaptive_var.s{i-1}.vgamma + (1 - parameter.beta1)*dgamma; 
            adaptive_var.s{i-1}.vbeta = parameter.beta1*adaptive_var.s{i-1}.vbeta + (1 - parameter.beta1)*dbeta;
            adaptive_var.V{i-1}.vgamma = max(parameter.beta2*adaptive_var.V{i-1}.vgamma + (1 - parameter.beta2)*dgamma.*dgamma,adaptive_var.V{i-1}.vgamma); 
            adaptive_var.V{i-1}.vbeta = max(parameter.beta2*adaptive_var.V{i-1}.vbeta + (1 - parameter.beta2)*dbeta.*dbeta,adaptive_var.V{i-1}.vbeta);
            
            
            dnn{i-1}.W = dnn{i-1}.W - parameter.learning_rate*(adaptive_var.s{i-1}.vW/(1-parameter.beta1.^global_step))./(parameter.delta + sqrt(adaptive_var.V{i-1}.vW./(1 - parameter.beta2.^global_step)));
            dnn{i-1}.b = dnn{i-1}.b - parameter.learning_rate*(adaptive_var.s{i-1}.vb/(1-parameter.beta1.^global_step))./(parameter.delta + sqrt(adaptive_var.V{i-1}.vb./(1 - parameter.beta2.^global_step)));
            dnn{i-1}.gamma = dnn{i-1}.gamma - parameter.learning_rate*(adaptive_var.s{i-1}.vgamma/(1-parameter.beta1.^global_step))./(parameter.delta + sqrt(adaptive_var.V{i-1}.vgamma./(1 - parameter.beta2.^global_step)));
            dnn{i-1}.beta = dnn{i-1}.beta - parameter.learning_rate*(adaptive_var.s{i-1}.vbeta/(1-parameter.beta1.^global_step))./(parameter.delta + sqrt(adaptive_var.V{i-1}.vbeta./(1 - parameter.beta2.^global_step)));
    end



    net.dnn = dnn;

end
% function [net, adaptive_var, loss] = train_DNN(net, parameter, adaptive_var, Y, label, global_step)
%     dnn = net.dnn;
%     L = size(dnn, 2) + 1;
%     y = Y.Y{L};
%     m = size(y, 2);
% 
%     disp(['Size of y: ', mat2str(size(y))]);
%     disp(['Size of label: ', mat2str(size(label))]);
%     loss = sum(sum(-label .* log(y))) / m;
%    
% 
%     初始化梯度
%     g = y - label;
% 
%     for i = L:-1:2
%         执行反向传播
%         if dnn{i-1}.function == "relu"
%             g = g .* (Y.Y{i} > 0);
%         elseif dnn{i-1}.function == "selu"
%             g = dselu(g, Y.Y{i});
%         elseif dnn{i-1}.function == "tanh"
%             g = dtanh(g, Y.Y{i});
%         elseif dnn{i-1}.function == "sigmoid"
%             g = g .* Y.Y{i} .* (1 - Y.Y{i});
%         elseif dnn{i-1}.function == "softmax"
%             g = Y.Y{i} - label;
%         end
%         
%         检查 Y.Cache 中的字段
%         if isfield(Y.Cache{i-1}, 'mu') && isfield(Y.Cache{i-1}, 'var')
%             dnn{i-1}.mu = Y.Cache{i-1}.mu;
%             dnn{i-1}.var = Y.Cache{i-1}.var;
%         else
%             disp(['mu or var not found in Y.Cache for layer ', num2str(i-1)]);
%         end
% 
%         反向传播求解梯度
%         [g, dgamma, dbeta] = batchnorm_backward(g, Y.Cache{i-1});
% 
%         更新参数
%         dw = g * Y.Y{i - 1}' / m;
%         db = sum(g, 2) / m;
%         g = dnn{i-1}.W' * g;
% 
%         更新参数的动量和 RMSProp 累积平方梯度
%         adaptive_var.s{i-1}.vW = parameter.beta1 * adaptive_var.s{i-1}.vW + (1 - parameter.beta1) * dw;
%         adaptive_var.s{i-1}.vb = parameter.beta1 * adaptive_var.s{i-1}.vb + (1 - parameter.beta1) * db;
%         adaptive_var.V{i-1}.vW = max(parameter.beta2 * adaptive_var.V{i-1}.vW + (1 - parameter.beta2) * dw .* dw, adaptive_var.V{i-1}.vW);
%         adaptive_var.V{i-1}.vb = max(parameter.beta2 * adaptive_var.V{i-1}.vb + (1 - parameter.beta2) * db .* db, adaptive_var.V{i-1}.vb);
%         
%         更新 gamma 和 beta 参数的动量和 RMSProp 累积平方梯度
%         adaptive_var.s{i-1}.vgamma = parameter.beta1 * adaptive_var.s{i-1}.vgamma + (1 - parameter.beta1) * dgamma;
%         adaptive_var.s{i-1}.vbeta = parameter.beta1 * adaptive_var.s{i-1}.vbeta + (1 - parameter.beta1) * dbeta;
%         adaptive_var.V{i-1}.vgamma = max(parameter.beta2 * adaptive_var.V{i-1}.vgamma + (1 - parameter.beta2) * dgamma .* dgamma, adaptive_var.V{i-1}.vgamma);
%         adaptive_var.V{i-1}.vbeta = max(parameter.beta2 * adaptive_var.V{i-1}.vbeta + (1 - parameter.beta2) * dbeta .* dbeta, adaptive_var.V{i-1}.vbeta);
% 
%         更新权重和偏置参数
%         net.dnn{i-1}.W = net.dnn{i-1}.W - parameter.learning_rate * (adaptive_var.s{i-1}.vW / (1 - parameter.beta1.^global_step)) ./ (parameter.delta + sqrt(adaptive_var.V{i-1}.vW ./ (1 - parameter.beta2.^global_step)));
%         net.dnn{i-1}.b = net.dnn{i-1}.b - parameter.learning_rate * (adaptive_var.s{i-1}.vb / (1 - parameter.beta1.^global_step)) ./ (parameter.delta + sqrt(adaptive_var.V{i-1}.vb ./ (1 - parameter.beta2.^global_step)));
%         net.dnn{i-1}.gamma = net.dnn{i-1}.gamma - parameter.learning_rate * (adaptive_var.s{i-1}.vgamma / (1 - parameter.beta1.^global_step)) ./ (parameter.delta + sqrt(adaptive_var.V{i-1}.vgamma ./ (1 - parameter.beta2.^global_step)));
%         net.dnn{i-1}.beta = net.dnn{i-1}.beta - parameter.learning_rate * (adaptive_var.s{i-1}.vbeta / (1 - parameter.beta1.^global_step)) ./ (parameter.delta + sqrt(adaptive_var.V{i-1}.vbeta ./ (1 - parameter.beta2.^global_step)));
%     end
% 
% end
