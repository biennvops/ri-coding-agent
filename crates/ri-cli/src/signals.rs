use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ShutdownSignal {
    Interrupt,
    Terminate,
    Hangup,
}

pub(crate) struct ShutdownSignals {
    receiver: mpsc::Receiver<ShutdownSignal>,
    stop: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<()>>,
}

impl ShutdownSignals {
    pub(crate) fn new() -> Self {
        let (sender, receiver) = mpsc::channel(1);
        let (stop, stop_receiver) = oneshot::channel();
        let task = tokio::spawn(listen_for_shutdown(sender, stop_receiver));
        Self {
            receiver,
            stop: Some(stop),
            task: Some(task),
        }
    }

    pub(crate) async fn recv(&mut self) -> Option<ShutdownSignal> {
        self.receiver.recv().await
    }

    #[cfg(test)]
    fn channel() -> (mpsc::Sender<ShutdownSignal>, Self) {
        let (sender, receiver) = mpsc::channel(1);
        (
            sender,
            Self {
                receiver,
                stop: None,
                task: None,
            },
        )
    }
}

impl Drop for ShutdownSignals {
    fn drop(&mut self) {
        let _ = self.stop.take().map(|stop| stop.send(()));
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

async fn listen_for_shutdown(
    sender: mpsc::Sender<ShutdownSignal>,
    mut stop: oneshot::Receiver<()>,
) {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};

        let Ok(mut interrupt) = signal(SignalKind::interrupt()) else {
            return;
        };
        let Ok(mut terminate) = signal(SignalKind::terminate()) else {
            return;
        };
        let Ok(mut hangup) = signal(SignalKind::hangup()) else {
            return;
        };

        tokio::select! {
            _ = &mut stop => {}
            signal = interrupt.recv() => {
                if signal.is_some() {
                    let _ = sender.send(ShutdownSignal::Interrupt).await;
                }
            }
            signal = terminate.recv() => {
                if signal.is_some() {
                    let _ = sender.send(ShutdownSignal::Terminate).await;
                }
            }
            signal = hangup.recv() => {
                if signal.is_some() {
                    let _ = sender.send(ShutdownSignal::Hangup).await;
                }
            }
        }
    }

    #[cfg(not(unix))]
    {
        tokio::select! {
            _ = &mut stop => {}
            result = tokio::signal::ctrl_c() => {
                if result.is_ok() {
                    let _ = sender.send(ShutdownSignal::Interrupt).await;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn injected_shutdown_signals_are_independently_testable() {
        let (sender, mut signals) = ShutdownSignals::channel();
        sender.send(ShutdownSignal::Terminate).await.unwrap();
        assert_eq!(signals.recv().await, Some(ShutdownSignal::Terminate));
    }

    #[tokio::test]
    async fn dropping_a_signal_source_does_not_keep_a_listener_alive() {
        let (_sender, signals) = ShutdownSignals::channel();
        drop(signals);
    }
}
