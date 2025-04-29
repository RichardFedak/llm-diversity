
window.app = new Vue({
    el: '#app',
    delimiters: ['[[', ']]'],
    data() {
      return {
        pairs: [],
        ratings: [],
        impl: "{{ impl }}",
        ratingOptions: [
          { value: null, text: 'Select a rating' },
          { value: 1, text: '1 - No diversity' },
          { value: 2, text: '2' },
          { value: 3, text: '3 - Moderate' },
          { value: 4, text: '4' },
          { value: 5, text: '5 - High diversity' }
        ]
      };
    },
    computed: {
      isFormValid() {
        return this.ratings.every(r => r !== null);
      }
    },
    async mounted() {
      const url = `${initial_data_url}?impl=${this.impl}`;
      const rawData = await fetch(url).then(resp => resp.json());
      const prepared = rawData.map(item => ({
        movieName: item.movie,
        movie: {
          idx: item.movie_idx,
          url: item.url
        }
      }));
      const pairs = [];
      for (let i = 0; i < prepared.length; i += 2) {
        const pair = [prepared[i], prepared[i + 1]];
        pairs.push(pair);
      }

      this.pairs = pairs;
      this.ratings = new Array(this.pairs.length).fill(null);
    },
    methods: {
      submit() {
        const form = document.createElement("form");
        form.method = "POST";
        form.action = continuation_url;
  
        // CSRF token
        const csrfInput = document.createElement("input");
        csrfInput.type = "hidden";
        csrfInput.name = "csrf_token";
        csrfInput.value = csrfToken;
        form.appendChild(csrfInput);
  
        // Include all diversity ratings
        this.ratings.forEach((rating, i) => {
          const input = document.createElement("input");
          input.type = "hidden";
          input.name = `rating_${i}`;
          input.value = rating;
          form.appendChild(input);
        });
  
        document.body.appendChild(form);
        form.submit();
      }
    }
  });