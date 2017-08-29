%% init

tau = 10; %ms
n_its = 1000;
n_in = 1;
pc = .1;
I0 = .001;
n_train_its = 50;
n_out = 1;

f = nan(n_its,n_in);
comp_set = rand(12,2);
for it = 1:n_its
    f(it,n_in) = sum(comp_set(:,2).*sin(comp_set(:,1)*.05*pi*it));
end

tic
for gradual_reduce = 1;
    for n_all = 100;
        for n_silent = [0 5 10 30];
            for begin_remove = [0];
                
                
                x = randn(n_all,1);
                r = tanh(x);
                dx = randn(n_all,1);
                ei = nan(n_train_its,n_all,n_its);
                eo = nan(n_train_its,n_out,n_its);
                Wr = randn(n_all)/sqrt(pc*n_all);
                Wi = randn(n_all,n_in);
                Wo = randn(n_all,n_out)/sqrt(n_all);
                r_train = nan(n_all,n_its);
                r_init = nan(n_all,n_its);
                R_all = nan(n_all,n_its);
                
                % Wfb = rand(N,1)*2-1;
                % n_plastic = round(.6*n_all);
                % i_plastic = sort(randperm(n_all,n_plastic));
                
                Pi_cell = cell(n_all,1);
                Po_cell_init = cell(n_out,1);
                for n = 1:n_all
                    Pi_cell{n} = eye(n_all)/(n_all);
                end
                for n = 1:n_out
                    Po_cell_init{n} = eye(n_all)/(n_all);
                end
                Po_cell = Po_cell_init;
                
                stim_dur = 50;
                out_dur = 50;
                train_window = 200;
                t_on = 200;
                t_off = t_on+stim_dur;
                t_end = t_off+train_window;
                y = zeros(n_its,n_in);
                y(t_on:t_off,1) = 5;
                
                %% record innate target
                for t = 1:n_its
                    for i = 1:n_all
                        dx(i) = 1/tau*(-x(i) + Wr(:,i)'*r + Wi(i,:)*y(t,:)');
                    end
                    x = x+dx;
                    r = tanh(x);
                    R_all(:,t) = r;
                end
                
                %% pretraining output
                I_noise = randn(n_all,n_its)*I0;
                for t = 1:n_its
                    for i = 1:n_all
                        dx(i) = 1/tau*(-x(i) + Wr(:,i)'*r + Wi(i,:)*y(t,:)' + I_noise(i,t));
                    end
                    x = x+dx;
                    r = tanh(x);
                    r_init(:,t) = r;
                end
                
                
                %% training
                for l = 1:n_train_its
                    I_noise = randn(n_all,n_its)*I0;
                    x = randn(n_all,1);
                    r = tanh(x);
                    
                    if n_train_its>begin_remove
                        %remove neurons gradually
                        [val, ind] = sort(abs(sum(R_all(:,t_on:t_end).^2,2)));
                        if gradual_reduce ==1
                            R_all(ind(1:min([n_silent l])),:) = 0; %silence neurons with the least effect on output
                        elseif gradual_reduce ==0
                            R_all(ind(1:n_silent),:) = 0; %silence neurons with the least effect on output
                        end
                    end
                    
                    for t = 1:t_end
                        r_train(:,t) = r;
                        
                        %innate training
                        ei(l,:,t) = r_train(:,t) - R_all(:,t);
                        
                        for i = 1:n_all
                            P = Pi_cell{i};
                            Pi_cell{i} = P - (P*r*r'*P)/(1+r'*P*r);
                            Wr(:,i) = Wr(:,i) - ei(l,i,t) * Pi_cell{i}*r;
                            
                            dx(i) = 1/tau*(-x(i) + Wr(:,i)'*r + Wi(i,:)*y(t,:)' + I_noise(i,t)); %+ Wfb(i)*z
                        end
                        x = x+dx;
                        r = tanh(x);
                        
                        eo(l,:,t) = Wo'*r - f(t);
                        for n = 1:n_out
                            Po = Po_cell_init{n};
                            Po_cell_init{n} = Po - (Po*r*r'*Po)/(1+r'*Po*r);
                            Wo(:,n) = Wo(:,n) - eo(l,n,t) * Po_cell_init{n}*r;
                        end
                    end
                end
                
                %% post train output
                n_reps = 100;
                I_noise = randn(n_all,n_its)*I0;
                r_test = nan(n_all,n_its,n_reps);
                for nr = 1:n_reps
                    x = randn(n_all,1);
                    r = tanh(x);
                    for t = 1:n_its
                        for i = 1:n_all
                            dx(i) = 1/tau*(-x(i) + Wr(:,i)'*r + Wi(i,:)*y(t,:)' + I_noise(i,t));
                        end
                        x = x+dx;
                        r = tanh(x);
                        z = Wo'*r;
                        r_test(:,t,nr) = r;
                    end
                end
                elapsedTime = toc;
                
                cd('/Users/Laura/code/rnns/data/complex')
                
                
                save(['ncells_' num2str(n_all) '_nsilent_' num2str(n_silent)...
                    '_greduce_' num2str(gradual_reduce) '_bremove_' num2str(begin_remove)])
            end
        end
    end
end