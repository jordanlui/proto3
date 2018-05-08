function plotArmPrediction(guessed,real,error)
    % Plot results
    
    subplot(2,2,1)
    hold on
    plot(guessed.wrist(:,1),'r:')
    plot(guessed.wrist(:,2),'m:')
    plot(guessed.wrist(:,3),'b:')
    plot(real.wrist(:,1),'r')
    plot(real.wrist(:,2),'m')
    plot(real.wrist(:,3),'b')
    hold off
    legend('x','y','z','x','y','z')
    title('Wrist real and predicted')

    subplot(2,2,2)
    plot(guessed.elbow(:,1),'r:'), hold on
    plot(guessed.elbow(:,2),'m:')
    plot(guessed.elbow(:,3),'b:')
    plot(real.elbow(:,1),'r')
    plot(real.elbow(:,2),'m')
    plot(real.elbow(:,3),'b')
    hold off
    legend('x','y','z','x','y','z')
    title('Elbow real and predicted')

    subplot(2,2,3)
    plot(error.wrist,'r.','LineStyle','none'), ylabel('Error (mm)')
    yyaxis right
    % hold on
    % scatter(1:length(guessed.wrist),guessWristdist,3,'m')
    % scatter(1:length(real.wrist),realWristDist,3,'b')
    % ylabel('Distance (mm)')
    % hold off
    title('Wrist Error')
%     legend('error','Prediction','real')

    subplot(2,2,4)
    plot(error.elbow,'r.','LineStyle','none'), ylabel('Error (mm)')
    yyaxis right
    % hold on
    % scatter(1:length(guessed.elbow),guessElbowdist,3,'m')
    % scatter(1:length(real.elbow),realElbowDist,3,'b')
    % hold off
    title('Elbow Error')
%     legend('error','Prediction','real')

end