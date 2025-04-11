
window.app = new Vue({
    el: '#app',
    methods: {
        onElicitationFinish(form) {
            this.busy = true;

            this.$el.querySelector('form').submit();
        }
    }
})