const row = document.querySelector('.scroll-row');
const wrapper = document.getElementById('scrollWrapper');
let isHovered = false;
let scrollTimer;

function pauseScroll() {
    isHovered = true;
    row.computedStyleMap.animationPlayState = 'paused';
}

function resumeScroll() {
    isHovered = false;
    row.style.animationPlayState = 'running';
}

wrapper.addEventListener('scroll', () => {
    clearTimeout(scrollTimer);
    pauseScroll();
    scrollTimer = setTimeout(() => {
        if (!isHovered) {
            resumeScroll();
        }
    }, 3000); // Adjust the timeout duration as needed
});