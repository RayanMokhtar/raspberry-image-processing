#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>

class ThreadPool {
public:
    // Constructeur avec nombre de threads par défaut = nombre de cœurs CPU
    ThreadPool(size_t numThreads = std::thread::hardware_concurrency());
    
    // Destruction du pool de threads
    ~ThreadPool();
    
    // Ajout d'une tâche au pool
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) 
        -> std::future<typename std::result_of<F(Args...)>::type>;
    
    // Obtenir le nombre de threads actifs
    size_t getThreadCount() const { return workers.size(); }
    
private:
    // Threads de travail
    std::vector<std::thread> workers;
    
    // File d'attente des tâches
    std::queue<std::function<void()>> tasks;
    
    // Synchronisation
    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop;
};

// Implémentation du template
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args) 
    -> std::future<typename std::result_of<F(Args...)>::type> {
    using return_type = typename std::result_of<F(Args...)>::type;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));
            
    std::future<return_type> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queueMutex);

        // Ne pas accepter de nouvelles tâches si le pool est arrêté
        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");

        tasks.emplace([task](){ (*task)(); });
    }
    condition.notify_one();
    return res;
}

#endif // THREAD_POOL_H