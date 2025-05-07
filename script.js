const row = document.querySelector('.scroll-row');
const wrapper = document.getElementById('scrollWrapper');
let isHovered = false;
let scrollTimer;

// Pause the scroll on hover
function pauseScroll() {
  isHovered = true;
  row.style.animationPlayState = 'paused';
}

// Resume the scroll on mouse leave or delay after scroll
function resumeScroll() {
  isHovered = false;
  row.style.animationPlayState = 'running';
}

// Resume scroll after 3s of no manual swipe
wrapper.addEventListener('scroll', () => {
  clearTimeout(scrollTimer);
  pauseScroll();
  scrollTimer = setTimeout(() => {
    if (!isHovered) resumeScroll();
  }, 3000);
});

// Clone the scroll-row to make it loop seamlessly
window.addEventListener('DOMContentLoaded', () => {
  const original = document.getElementById('scrollRow');
  const clone = original.cloneNode(true);
  clone.setAttribute('aria-hidden', 'true'); // for accessibility
  original.parentNode.appendChild(clone);
});
