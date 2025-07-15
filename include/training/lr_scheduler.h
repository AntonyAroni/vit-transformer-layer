#pragma once

class LRScheduler {
private:
    double initial_lr;
    double current_lr;
    int warmup_steps;
    int total_steps;
    int current_step;

public:
    LRScheduler(double lr, int warmup = 1000, int total = 10000);
    double getNextLR();
    double getCurrentLR() const { return current_lr; }
};