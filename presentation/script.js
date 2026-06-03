function updateScale() {
  var margin = 80;
  var scale = Math.min((window.innerWidth - margin) / 1920, 1);
  document.documentElement.style.setProperty('--slide-scale', scale);
}

function scrollToSlide(direction) {
  var slides = Array.prototype.slice.call(document.querySelectorAll('.slide-outer'));
  if (!slides.length) {
    return;
  }

  var currentY = window.scrollY;
  var currentIndex = 0;
  var smallestDistance = Infinity;

  slides.forEach(function (slide, index) {
    var distance = Math.abs(slide.offsetTop - currentY);
    if (distance < smallestDistance) {
      smallestDistance = distance;
      currentIndex = index;
    }
  });

  var nextIndex = Math.max(0, Math.min(slides.length - 1, currentIndex + direction));
  slides[nextIndex].scrollIntoView({ behavior: 'smooth', block: 'start' });
}

document.addEventListener('DOMContentLoaded', function () {
  updateScale();
  window.addEventListener('resize', updateScale);

  var printButton = document.querySelector('.print-button');
  if (printButton) {
    printButton.addEventListener('click', function () {
      window.print();
    });
  }

  window.addEventListener('keydown', function (event) {
    if (event.key === 'ArrowDown' || event.key === 'PageDown' || event.key === 'ArrowRight') {
      event.preventDefault();
      scrollToSlide(1);
    }

    if (event.key === 'ArrowUp' || event.key === 'PageUp' || event.key === 'ArrowLeft') {
      event.preventDefault();
      scrollToSlide(-1);
    }
  });
});
