
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
      this.pairs = rawData.map(item => ({
        movies: item.pair,  // [movie1, movie2]
        version: item.version
      }));
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
  
        this.pairs.forEach((pair, i) => {
          const inputRating = document.createElement("input");
          inputRating.type = "hidden";
          inputRating.name = `rating_${i}`;
          inputRating.value = this.ratings[i];
          form.appendChild(inputRating);
        
          const inputVersion = document.createElement("input");
          inputVersion.type = "hidden";
          inputVersion.name = `version_${i}`;
          inputVersion.value = pair.version;
          form.appendChild(inputVersion);
        });        
  
        document.body.appendChild(form);
        form.submit();
      }
    }
  });