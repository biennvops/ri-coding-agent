use std::time::{Duration, Instant};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RedrawUrgency {
    Immediate,
    Coalesced,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RedrawScheduler {
    coalesce_window: Duration,
    immediate: bool,
    deadline: Option<Instant>,
}

impl RedrawScheduler {
    pub fn new(coalesce_window: Duration) -> Self {
        Self {
            coalesce_window,
            immediate: false,
            deadline: None,
        }
    }

    pub fn request(&mut self, urgency: RedrawUrgency, now: Instant) {
        match urgency {
            RedrawUrgency::Immediate => {
                self.immediate = true;
                self.deadline = None;
            }
            RedrawUrgency::Coalesced if !self.immediate => {
                self.deadline
                    .get_or_insert_with(|| now + self.coalesce_window);
            }
            RedrawUrgency::Coalesced => {}
        }
    }

    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    pub fn is_pending(&self) -> bool {
        self.immediate || self.deadline.is_some()
    }

    pub fn mark_drawn(&mut self) {
        self.immediate = false;
        self.deadline = None;
    }

    pub fn take_ready(&mut self, now: Instant) -> bool {
        let ready = self.immediate || self.deadline.is_some_and(|deadline| now >= deadline);
        if ready {
            self.immediate = false;
            self.deadline = None;
        }
        ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settled_scheduler_has_no_idle_deadline() {
        let now = Instant::now();
        let mut scheduler = RedrawScheduler::new(Duration::from_millis(12));
        assert!(!scheduler.is_pending());
        scheduler.request(RedrawUrgency::Immediate, now);
        assert!(scheduler.take_ready(now));
        assert!(!scheduler.is_pending());
        assert_eq!(scheduler.deadline(), None);
    }

    #[test]
    fn coalesced_requests_share_one_bounded_deadline() {
        let now = Instant::now();
        let mut scheduler = RedrawScheduler::new(Duration::from_millis(12));
        for offset in 0..1_000 {
            scheduler.request(
                RedrawUrgency::Coalesced,
                now + Duration::from_micros(offset),
            );
        }
        let deadline = scheduler.deadline().expect("burst should arm a deadline");
        assert_eq!(deadline, now + Duration::from_millis(12));
        assert!(!scheduler.take_ready(now + Duration::from_millis(11)));
        assert!(scheduler.take_ready(deadline));
        assert_eq!(scheduler.deadline(), None);
    }

    #[test]
    fn immediate_requests_bypass_a_pending_coalescing_window() {
        let now = Instant::now();
        let mut scheduler = RedrawScheduler::new(Duration::from_millis(12));
        scheduler.request(RedrawUrgency::Coalesced, now);
        scheduler.request(RedrawUrgency::Immediate, now);
        assert!(scheduler.take_ready(now));
        assert_eq!(scheduler.deadline(), None);
    }
}
